import json, requests, traceback
import numpy as np, pandas as pd
from scipy import signal as scisig
from datetime import datetime, timezone
from io import BytesIO

WINDOW_SIZE_SEC, WINDOW_OVERLAP = 60, 0.5
E4_OFFICIAL_SAMPLING_RATE = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "HR": 1}
FILTER_CONFIG = {
    "EDA": {"type": "lowpass", "cutoff": 0.5, "order": 4},
    "BVP": {"type": "bandpass", "cutoff": [0.5, 8], "order": 4},
    "ACC": {"type": "lowpass", "cutoff": 5, "order": 3},
    "TEMP": {"type": "lowpass", "cutoff": 0.1, "order": 3}
}

def read_e4_csv(file_content, file_name):
    file_type = file_name.replace(".csv", "").upper()
    df = pd.read_csv(BytesIO(file_content), header=None)

    if file_type == "IBI":
        start_ts, data = df.iloc[0, 0], df.iloc[1:].values
        return {"data": data[:, 1], "time_axis": np.array([datetime.fromtimestamp(start_ts + t, tz=timezone.utc) for t in data[:, 0]]), 
                "relative_time": data[:, 0], "start_timestamp": start_ts, "file_type": file_type}
    
    if file_type == "TAGS":
        timestamps = df.iloc[:, 0].values
        return {"data": timestamps, "time_axis": np.array([datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]), "file_type": file_type}

    start_ts, sr = df.iloc[0, 0], df.iloc[1, 0]
    raw_data = df.iloc[2:].values
    data = raw_data / 64 if file_type == "ACC" else raw_data.flatten()
    num_samples = len(data)
    time_rel = np.linspace(0, num_samples / sr, num_samples)
    return {
        "data": data, "time_axis": np.array([datetime.fromtimestamp(start_ts + t, tz=timezone.utc) for t in time_rel]),
        "sampling_rate": sr, "start_timestamp": start_ts, "end_timestamp": start_ts + num_samples / sr, "file_type": file_type
    }

def butterworth_filter(signal, fs, filter_type, cutoff, order=4):
    nyquist = fs * 0.5
    norm_cutoff = [c / nyquist for c in cutoff] if isinstance(cutoff, list) else cutoff / nyquist
    b, a = scisig.butter(order, norm_cutoff, btype=filter_type, analog=False)
    return scisig.filtfilt(b, a, signal)

def extract_stat_features(sig):
    if len(sig) == 0 or np.all(np.isnan(sig)): return {}
    return {"mean": float(np.nanmean(sig)), "std": float(np.nanstd(sig)), "max": float(np.nanmax(sig)), 
            "min": float(np.nanmin(sig)), "range": float(np.nanmax(sig) - np.nanmin(sig)), "median": float(np.nanmedian(sig)),
            "rms": float(np.sqrt(np.nanmean(sig ** 2))), "skewness": float(pd.Series(sig).skew()), "kurtosis": float(pd.Series(sig).kurtosis())}

def extract_eda_features(win, fs):
    stat = extract_stat_features(win)
    thresh = np.mean(win) + np.std(win)
    peaks, props = scisig.find_peaks(win, height=thresh, distance=fs)
    stat.update({"scr_peak_count": len(peaks), "scr_peak_mean_height": float(np.mean(props['peak_heights']) if len(peaks) > 0 else 0),
                 "scr_peak_max_height": float(np.max(props['peak_heights']) if len(peaks) > 0 else 0)})
    return stat

def extract_acc_features(win_3d):
    feat = {}
    for i, axis in enumerate(["x", "y", "z"]):
        af = extract_stat_features(win_3d[:, i])
        feat.update({f"{axis}_{k}": v for k, v in af.items()})
    mag = np.sqrt(np.sum(win_3d ** 2, axis=1))
    mf = extract_stat_features(mag)
    feat.update({f"mag_{k}": v for k, v in mf.items()})
    feat["activity_level"] = "静息" if mf["mean"] < 1.05 else "低活动" if mf["mean"] < 1.2 else "中高活动"
    return feat

def extract_hrv_features(ibi):
    ibi_ms = ibi * 1000
    nn = ibi_ms[np.where((ibi_ms > 300) & (ibi_ms < 2000))[0]]
    if len(nn) < 5: return {"error": "有效IBI数据不足，无法计算HRV特征"}
    
    diff_nn = np.abs(np.diff(nn))
    nn50 = len(diff_nn[diff_nn > 50])
    feat = {"hrv_sdnn": float(np.std(nn)), "hrv_rmssd": float(np.sqrt(np.mean(diff_nn ** 2))), "hrv_nn50": int(nn50),
            "hrv_pnn50": float(nn50 / len(diff_nn) * 100), "mean_ibi_ms": float(np.mean(nn)), "mean_hr_bpm": float(60000 / np.mean(nn))}
    try:
        f, pxx = scisig.welch(nn, fs=1/np.mean(ibi), nperseg=min(256, len(nn)))
        lf, hf = np.trapz(pxx[(f >= 0.04) & (f <= 0.15)], f[(f >= 0.04) & (f <= 0.15)]), np.trapz(pxx[(f >= 0.15) & (f <= 0.4)], f[(f >= 0.15) & (f <= 0.4)])
        feat.update({"hrv_lf_power": float(lf), "hrv_hf_power": float(hf), "hrv_lf_hf_ratio": float(lf / hf if hf > 0 else 0)})
    except: feat["hrv_freq_note"] = "频域特征计算失败，数据长度不足"
    return feat


def sliding_window_feature_extraction(processed_data, ibi_data=None, tags_data=None):
    req_keys = ["EDA", "BVP", "ACC", "TEMP", "HR"]
    if not all(k in processed_data for k in req_keys): raise ValueError(f"缺少必要的键")
    
    base_time, base_fs = processed_data["EDA"]["time_axis"], E4_OFFICIAL_SAMPLING_RATE["EDA"]
    win_samps, step_samps, total_len = int(WINDOW_SIZE_SEC * base_fs), int(WINDOW_SIZE_SEC * base_fs * (1 - WINDOW_OVERLAP)), len(base_time)
    dur = (base_time[-1] - base_time[0]).total_seconds() / 3600
    
    output = {"data_basic_info": {"start_time": base_time[0].isoformat(), "end_time": base_time[-1].isoformat(),
                                  "total_duration_hours": round(dur, 2), "window_size_sec": WINDOW_SIZE_SEC, "window_overlap": WINDOW_OVERLAP},
              "data_quality_check": {}, "global_health_features": {}, "time_series_windows": [], "event_tags": []}

    for mod in processed_data:
        try:
            d_len, sr = len(processed_data[mod]["data"]), processed_data[mod].get("sampling_rate", E4_OFFICIAL_SAMPLING_RATE.get(mod, 1))
            s_ts, e_ts = processed_data[mod].get("start_timestamp", base_time[0].timestamp()), processed_data[mod].get("end_timestamp", base_time[-1].timestamp())
            exp_len = int((e_ts - s_ts) * sr)
            output["data_quality_check"][mod] = {"total_samples": int(d_len), "missing_rate": round((1 - d_len / exp_len) * 100, 2) if exp_len > 0 else 0, "sampling_rate": float(sr)}
        except Exception as e:
            output["data_quality_check"][mod] = {"error": str(e), "total_samples": len(processed_data[mod].get("data", [])), "sampling_rate": float(E4_OFFICIAL_SAMPLING_RATE.get(mod, 1))}

    def _safe_extract(feat_func, data, *args):
        try: return feat_func(data, *args)
        except Exception as e: return {"error": str(e)}

    try:
        if ibi_data and len(ibi_data.get("data", [])) > 0: output["global_health_features"]["hrv_full_period"] = extract_hrv_features(ibi_data["data"])
    except Exception as e: output["global_health_features"]["hrv_full_period"] = {"error": str(e)}
    
    try:
        if "HR" in processed_data and len(processed_data["HR"].get("data", [])) > 0: output["global_health_features"]["hr_full_period"] = extract_stat_features(processed_data["HR"]["data"])
    except Exception as e: output["global_health_features"]["hr_full_period"] = {"error": str(e)}
    
    try:
        if tags_data and len(tags_data.get("data", [])) > 0: output["event_tags"] = [{"event_id": i, "timestamp": ts.isoformat()} for i, ts in enumerate(tags_data["time_axis"])]
    except Exception as e: output["event_tags"] = [{"error": str(e)}]

    i = 0
    while i + win_samps < total_len:
        try:
            start_idx, end_idx = i, i + win_samps
            start_time, end_time = base_time[start_idx], base_time[end_idx]
            win_res = {"window_id": len(output["time_series_windows"]), "window_start_time": start_time.isoformat(), "window_end_time": end_time.isoformat()}
            
            win_res["eda_features"] = _safe_extract(extract_eda_features, processed_data["EDA"]["filtered_data"][start_idx:end_idx], base_fs)
            
            for mod, key in [("TEMP", "temp_features"), ("BVP", "bvp_features")]:
                t_axis = processed_data[mod]["time_axis"]
                mask = (t_axis >= start_time) & (t_axis <= end_time)
                win_res[key] = _safe_extract(extract_stat_features, processed_data[mod]["filtered_data"][mask])
            
            acc_axis = processed_data["ACC"]["time_axis"]
            acc_mask = (acc_axis >= start_time) & (acc_axis <= end_time)
            win_res["acc_features"] = _safe_extract(extract_acc_features, processed_data["ACC"]["filtered_data"][acc_mask])
            
            hr_axis = processed_data["HR"]["time_axis"]
            hr_mask = (hr_axis >= start_time) & (hr_axis <= end_time)
            win_res["hr_features"] = _safe_extract(extract_stat_features, processed_data["HR"]["data"][hr_mask])
            
            if ibi_data and len(ibi_data.get("data", [])) > 0:
                ibi_mask = (ibi_data["time_axis"] >= start_time) & (ibi_data["time_axis"] <= end_time)
                ibi_win = ibi_data["data"][ibi_mask]
                win_res["hrv_features"] = _safe_extract(extract_hrv_features, ibi_win) if len(ibi_win) >= 5 else {"note": "窗口内IBI数据不足"}
            
            output["time_series_windows"].append(win_res)
        except Exception as e:
            output["time_series_windows"].append({"window_id": len(output["time_series_windows"]), "error": str(e)})
        i += step_samps
    
    return output

def main(e4_csv_files=None, sys_files=None, **kwargs):
    """
    主函数 - 支持多种调用方式
    
    参数:
        e4_csv_files: 自定义文件列表参数
        sys_files: Dify 系统变量 sys.files
        **kwargs: 其他参数（包括可能的 sys 字典）
    """
    import os
    
    # 优先级：sys_files > sys.files（来自kwargs） > e4_csv_files
    files_to_process = None
    
    if sys_files:
        files_to_process = sys_files
    elif "sys" in kwargs and isinstance(kwargs["sys"], dict):
        files_to_process = kwargs["sys"].get("files", [])
    elif e4_csv_files:
        files_to_process = e4_csv_files
    
    if not files_to_process:
        return {"success": False, "error": "未接收到上传的CSV文件。在 Dify 中使用：main(e4_csv_files=e4_csv_files)", "feature_json_str": ""}
    
    # 按官方文档规则设置文件 URL 基础地址
    # 优先级：FILES_URL > CONSOLE_API_URL > SERVICE_API_URL > 默认值
    files_url = os.environ.get("FILES_URL", "").rstrip('/')
    console_api_url = os.environ.get("CONSOLE_API_URL", "").rstrip('/')
    service_api_url = os.environ.get("SERVICE_API_URL", "").rstrip('/')
    app_api_url = os.environ.get("APP_API_URL", "").rstrip('/')
    dify_port = os.environ.get("DIFY_PORT", "5001")
    
    # 基础 URL：优先使用 FILES_URL，如果都不存在则使用默认值（带端口）
    dify_base_url = files_url or console_api_url or service_api_url or app_api_url or f"http://127.0.0.1:{dify_port}"
    
    # 确保 URL 包含正确的端口（如果没有指定端口，添加默认端口）
    if dify_base_url and ":" not in dify_base_url.split("//", 1)[-1]:  # 检查是否有端口
        dify_base_url = f"{dify_base_url}:{dify_port}"
    
    # 对于内部服务通信（Docker 网络内），优先使用 INTERNAL_FILES_URL
    # 如果为空，尝试使用 Docker 服务名 'api'，再回退到外部 URL
    internal_files_url = os.environ.get("INTERNAL_FILES_URL", "").rstrip('/')
    if internal_files_url:
        dify_internal_url = internal_files_url
    else:
        # 尝试使用 Docker 服务名（仅在可能的 Docker 网络中）
        dify_internal_url = f"http://api:{dify_port}"
    
    file_storage = {t: None for t in ["EDA", "BVP", "ACC", "TEMP", "HR", "IBI", "TAGS"]}
    req_types = ["EDA", "BVP", "ACC", "TEMP", "HR"]
    file_type_keywords = {"EDA": "eda", "BVP": "bvp", "ACC": "acc", "TEMP": "temp", "HR": "hr", "IBI": "ibi", "TAGS": ["tags", "tag"]}
    
    def _download_file(file_obj):
        """从文件对象中提取文件名和内容"""
        if not isinstance(file_obj, dict):
            return None, None
        
        filename = file_obj.get("filename", "").lower()
        if not filename:
            return None, None
        
        fcontent = None
        
        # 优先级1: 直接的 URL 字段（file-preview 或 download）
        for url_key in ["url", "download_url", "remote_url"]:
            url = file_obj.get(url_key, "")
            if not url:
                continue
            
            # 构建完整 URL 列表（优先使用内部 URL，再尝试外部 URL）
            urls_to_try = []
            
            if url.startswith("http"):
                # 已经是完整 URL，直接使用
                urls_to_try.append(url)
            else:
                # 相对路径：尝试多个策略
                # 1. 先尝试 Docker 内部 API 服务名
                urls_to_try.append(dify_internal_url + url)
                # 2. 再尝试外部基础 URL
                if dify_internal_url != dify_base_url:
                    urls_to_try.append(dify_base_url + url)
                # 3. 最后尝试带完整端口的默认地址
                urls_to_try.append(f"http://127.0.0.1:{dify_port}{url}")
            
            for full_url in urls_to_try:
                try:
                    resp = requests.get(full_url, timeout=10)
                    if resp.status_code == 200 and len(resp.content) > 0:
                        fcontent = resp.content
                        return filename, fcontent
                except:
                    continue  # 继续尝试下一个 URL
        
        # 优先级2: 直接的文件内容
        if "content" in file_obj:
            try:
                content = file_obj.get("content")
                if isinstance(content, str):
                    fcontent = content.encode()
                elif isinstance(content, bytes):
                    fcontent = content
                
                if fcontent and len(fcontent) > 0:
                    return filename, fcontent
            except:
                pass
        
        # 优先级3: 通过 related_id 访问（仅作为备选）
        related_id = file_obj.get("related_id", "")
        if related_id:
            api_endpoints = [
                f"{dify_internal_url}/v1/files/{related_id}/download",
                f"{dify_internal_url}/v1/files/{related_id}",
                f"{dify_base_url}/v1/files/{related_id}/download",
                f"{dify_base_url}/v1/files/{related_id}",
            ]
            
            for endpoint in api_endpoints:
                try:
                    resp = requests.get(endpoint, timeout=10)
                    if resp.status_code == 200 and len(resp.content) > 0:
                        fcontent = resp.content
                        return filename, fcontent
                except:
                    continue
        
        # 如果都失败，返回文件名但内容为 None
        return filename, None
    
    # 处理文件列表
    processed_files = []
    for f in files_to_process:
        fname, fcontent = _download_file(f)
        processed_files.append({"filename": fname or "unknown", "size": len(fcontent) if fcontent else 0, "success": bool(fcontent)})
        
        if not fname or not fname.endswith(".csv") or not fcontent:
            continue
        
        for file_type, keywords in file_type_keywords.items():
            kw_list = keywords if isinstance(keywords, list) else [keywords]
            if any(k in fname for k in kw_list):
                file_storage[file_type] = {"content": fcontent, "name": fname}
                break
    
    missing = [t for t in req_types if file_storage[t] is None]
    if missing:
        debug_info = {"processed_files": processed_files, "missing": missing}
        return {"success": False, "error": f"缺少必须的文件：{','.join(missing)}。调试信息：{json.dumps(debug_info, ensure_ascii=False)}", "feature_json_str": ""}
    
    try:
        raw_data = {t: read_e4_csv(file_storage[t]["content"], file_storage[t]["name"]) for t in req_types}
        opt_data = {}
        if file_storage["IBI"]: 
            opt_data["IBI"] = read_e4_csv(file_storage["IBI"]["content"], file_storage["IBI"]["name"])
        if file_storage["TAGS"]: 
            opt_data["TAGS"] = read_e4_csv(file_storage["TAGS"]["content"], file_storage["TAGS"]["name"])
    except Exception as e:
        return {"success": False, "error": f"CSV文件读取失败：{str(e)}\n详情：{traceback.format_exc()}", "feature_json_str": ""}
    
    try:
        proc_data = {}
        for mod in raw_data:
            mod_data = raw_data[mod].copy()
            if mod in FILTER_CONFIG:
                cfg = FILTER_CONFIG[mod]
                fs = mod_data["sampling_rate"]
                if mod == "ACC":
                    filt = np.zeros_like(mod_data["data"])
                    for i in range(3): filt[:, i] = butterworth_filter(mod_data["data"][:, i], fs, cfg["type"], cfg["cutoff"], cfg["order"])
                else:
                    filt = butterworth_filter(mod_data["data"], fs, cfg["type"], cfg["cutoff"], cfg["order"])
                mod_data["filtered_data"] = filt
            else:
                mod_data["filtered_data"] = mod_data["data"]
            proc_data[mod] = mod_data
    except Exception as e:
        return {"success": False, "error": f"信号预处理滤波失败：{str(e)}\n详情：{traceback.format_exc()}", "feature_json_str": ""}
    
    try:
        feat_json = sliding_window_feature_extraction(proc_data, ibi_data=opt_data.get("IBI"), tags_data=opt_data.get("TAGS"))
        
        # 首先尝试带缩进的 JSON（更易读）
        feat_str = json.dumps(feat_json, ensure_ascii=False, indent=2)
        
        # 检查长度限制（Dify 限制：400000 字符）
        if len(feat_str) > 400000:
            # 尝试压缩：删除缩进
            feat_str = json.dumps(feat_json, ensure_ascii=False, separators=(',', ':'))
            
            if len(feat_str) > 400000:
                # 如果仍然超过限制，尝试减少数据
                # 保留最后 N 个窗口（减半）
                if len(feat_json.get("time_series_windows", [])) > 2:
                    feat_json["time_series_windows"] = feat_json["time_series_windows"][::2]  # 只保留偶数索引的窗口
                    feat_str = json.dumps(feat_json, ensure_ascii=False, separators=(',', ':'))
                
                if len(feat_str) > 400000:
                    # 如果还是太大，返回错误
                    return {
                        "success": False, 
                        "error": f"特征JSON过大（{len(feat_str)} 字符），超过 Dify 限制（400000 字符）。请减少数据量或增加窗口重叠比例。", 
                        "feature_json_str": ""
                    }
        
        if not feat_str or len(feat_str.strip()) == 0:
            return {
                "success": False, 
                "error": "特征JSON转换为空字符串", 
                "feature_json_str": ""
            }
    except Exception as e:
        return {
            "success": False, 
            "error": f"特征提取失败：{str(e)}\n详情：{traceback.format_exc()}", 
            "feature_json_str": ""
        }
    
    return {"success": True, "feature_json_str": feat_str, "error": ""}

# ==========================================
# Dify 工作流使用说明（基于 docker-compose 配置）
# ==========================================
# 
# 【环境变量配置】在 Dify docker-compose.yaml 中设置：
# 
# 方案1：推荐（最安全）
#   FILES_URL: http://127.0.0.1:5001
#   
# 方案2：使用控制台 API URL
#   CONSOLE_API_URL: http://127.0.0.1:5001
#   
# 方案3：使用服务 API URL  
#   SERVICE_API_URL: http://127.0.0.1:5001/v1
#
# 方案4：Docker 内部通信（如果在 Dify 容器内运行代码）
#   INTERNAL_FILES_URL: http://api:5001
#
# 若都未设置，默认使用：http://127.0.0.1:5001
#
# 【代码节点调用示例】
# 
# 方式1：自动检测（推荐）
#   result = auto_main(**locals())
#
# 方式2：显式指定文件变量
#   result = main(e4_csv_files=files)
#   # 如果你的文件变量叫 "files" "documents" 或其他名称
#
# 【输出限制】
# 
# - feature_json_str 长度限制：< 400000 字符（Dify 限制）
# - 代码会自动：
#   1. 首先用缩进格式生成 JSON（易读）
#   2. 如果超过 400000 字符，自动压缩为紧凑格式
#   3. 如果仍超过，自动下采样窗口（保留偶数索引）
#   4. 如果还是超过，返回错误并建议减少数据量
#
# 【优化建议】
# 
# 如果输出超过 400000 字符限制，可以：
# 1. 增加 WINDOW_OVERLAP（当前 50%）→ 减少窗口数量
# 2. 增加 WINDOW_SIZE_SEC（当前 60秒）→ 输出更少的窗口
# 3. 减少输入数据时间长度
# 4. 使用 diagnose_urls(**locals()) 检查数据量
# ==========================================

# 自动检测版本 - 用于 Dify 工作流
def auto_main(**all_vars):
    """自动从 Dify 传递的所有变量中查找文件列表"""
    import os
    
    # 常见的文件变量名
    common_names = ["files", "file_list", "documents", "e4_csv_files", "csv_files", "uploaded_files"]
    
    files_to_process = None
    
    # 1. 先查找常见的文件变量名
    for var_name in common_names:
        if var_name in all_vars and all_vars[var_name]:
            files_to_process = all_vars[var_name]
            break
    
    # 2. 如果没找到，查找任何列表类型的变量（可能是文件）
    if not files_to_process:
        for var_name, var_value in all_vars.items():
            if isinstance(var_value, list) and len(var_value) > 0:
                # 检查是否是文件对象
                if isinstance(var_value[0], dict) and any(k in var_value[0] for k in ["filename", "url", "related_id"]):
                    files_to_process = var_value
                    break
    
    if not files_to_process:
        return {"success": False, "error": "自动检测失败：未找到文件列表。请在代码节点中明确指定：result = main(e4_csv_files=your_file_variable)", "feature_json_str": ""}
    
    return main(e4_csv_files=files_to_process)


def diagnose_urls(**all_vars):
    """诊断当前的 URL 配置和可用的文件列表"""
    import os
    
    diag = {
        "environment_variables": {
            "FILES_URL": os.environ.get("FILES_URL", "[未设置]"),
            "CONSOLE_API_URL": os.environ.get("CONSOLE_API_URL", "[未设置]"),
            "SERVICE_API_URL": os.environ.get("SERVICE_API_URL", "[未设置]"),
            "APP_API_URL": os.environ.get("APP_API_URL", "[未设置]"),
            "INTERNAL_FILES_URL": os.environ.get("INTERNAL_FILES_URL", "[未设置]"),
            "DIFY_PORT": os.environ.get("DIFY_PORT", "5001"),
        },
        "detected_variables": [],
        "file_lists_found": {}
    }
    
    # 检测所有列表类型的变量
    for var_name, var_value in all_vars.items():
        if isinstance(var_value, list) and len(var_value) > 0:
            diag["detected_variables"].append({
                "name": var_name,
                "type": "list",
                "length": len(var_value),
                "is_file_list": isinstance(var_value[0], dict) and any(k in var_value[0] for k in ["filename", "url", "related_id"])
            })
            
            # 如果是文件列表，记录详情
            if isinstance(var_value[0], dict) and any(k in var_value[0] for k in ["filename", "url", "related_id"]):
                files_info = []
                for f in var_value[:3]:  # 只显示前3个文件
                    files_info.append({
                        "filename": f.get("filename", "N/A"),
                        "has_url": "url" in f,
                        "has_content": "content" in f,
                        "has_related_id": "related_id" in f,
                    })
                diag["file_lists_found"][var_name] = files_info
    
    return diag


def estimate_output_size(num_windows=100, with_ibi=True, with_tags=False):
    """
    估计输出 JSON 大小（单位：字符）
    
    参数:
        num_windows: 预期的滑动窗口数量
        with_ibi: 是否包含 IBI/HRV 特征
        with_tags: 是否包含标签数据
    
    返回:
        估计的字符数
    """
    # 每个窗口的基础特征大小（粗略估计）
    base_window_size = 800  # 字符
    hrv_size = 200 if with_ibi else 0
    tags_size = 150 * (num_windows // 10) if with_tags else 0  # 平均每10个窗口有个标签
    
    # 全局信息大小
    global_overhead = 500
    
    # 总计算
    total = global_overhead + (base_window_size + hrv_size) * num_windows + tags_size
    
    return total


def optimize_for_dify_limit(duration_hours=1, sampling_rate_eda=4):
    """
    根据输入数据时长建议优化的窗口参数，以保持输出在 400000 字符以内
    
    参数:
        duration_hours: 输入数据的时长（小时）
        sampling_rate_eda: EDA 采样率（Hz，默认 4Hz）
    
    返回:
        建议的参数配置
    """
    # 当前配置
    current_window_size = WINDOW_SIZE_SEC  # 60 秒
    current_overlap = WINDOW_OVERLAP  # 50%
    
    # 计算当前窗口数量
    duration_sec = duration_hours * 3600
    step_sec = current_window_size * (1 - current_overlap)
    num_windows = int((duration_sec - current_window_size) / step_sec) + 1
    
    # 估计当前输出大小
    current_estimate = estimate_output_size(num_windows=num_windows, with_ibi=True, with_tags=False)
    
    recommendation = {
        "input_duration_hours": duration_hours,
        "current_window_size_sec": current_window_size,
        "current_overlap": current_overlap,
        "estimated_num_windows": num_windows,
        "estimated_output_chars": current_estimate,
        "exceeds_limit": current_estimate > 400000,
        "suggestions": []
    }
    
    if current_estimate > 400000:
        # 提供优化建议
        factor = current_estimate / 400000
        
        # 方案1：增加窗口大小
        suggested_window = int(current_window_size * (factor ** 0.5))
        recommendation["suggestions"].append({
            "method": "增加窗口大小",
            "recommended_value": suggested_window,
            "param": f"WINDOW_SIZE_SEC = {suggested_window}",
            "new_num_windows": int((duration_sec - suggested_window) / (suggested_window * (1 - current_overlap))) + 1
        })
        
        # 方案2：增加重叠比例
        suggested_overlap = 1 - (current_window_size * (1 - current_overlap) * factor)
        suggested_overlap = min(0.9, max(0, suggested_overlap))
        recommendation["suggestions"].append({
            "method": "增加窗口重叠",
            "recommended_value": suggested_overlap,
            "param": f"WINDOW_OVERLAP = {suggested_overlap:.2f}",
            "new_num_windows": int((duration_sec - current_window_size) / (current_window_size * (1 - suggested_overlap))) + 1
        })
        
        # 方案3：减少输入数据时长
        suggested_duration = duration_hours / factor
        recommendation["suggestions"].append({
            "method": "减少输入数据时长",
            "recommended_value": suggested_duration,
            "param": f"input_duration ≈ {suggested_duration:.2f} 小时",
            "new_num_windows": int(suggested_duration * 3600 / (current_window_size * (1 - current_overlap)))
        })
    else:
        recommendation["suggestions"].append({
            "status": "符合限制",
            "message": f"当前配置下输出大小 ({current_estimate} 字符) 远低于限制 (400000 字符)"
        })
    
    return recommendation