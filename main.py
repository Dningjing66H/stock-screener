import os, sys, json, time, hmac, hashlib, base64, traceback
from datetime import datetime, timedelta
import requests
import pandas as pd
import akshare as ak

FEISHU_WEBHOOK = os.environ.get("FEISHU_WEBHOOK", "")
FEISHU_SECRET  = os.environ.get("FEISHU_SECRET", "")
LLM_API_KEY    = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_URL        = os.environ.get("LLM_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL      = os.environ.get("LLM_MODEL", "deepseek-chat")
MAX_ENRICH     = int(os.environ.get("MAX_ENRICH", "60"))
RETRY_MODE     = os.environ.get("RETRY_MODE", "0") == "1"

NOW = datetime.now()
TODAY_STR = NOW.strftime("%Y%m%d")
START_KLINE = (NOW - timedelta(days=150)).strftime("%Y%m%d")

def log(*a): print(*a, flush=True)

def feishu_sign(secret: str):
    ts = str(int(time.time()))
    string_to_sign = f"{ts}\n{secret}"
    h = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return ts, base64.b64encode(h).decode("utf-8")

def push_feishu(title: str, markdown: str):
    if not FEISHU_WEBHOOK:
        log("未配置 FEISHU_WEBHOOK，跳过推送"); return
    body = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}},
            "elements": [{"tag": "markdown", "content": markdown}],
        },
    }
    if FEISHU_SECRET:
        ts, sign = feishu_sign(FEISHU_SECRET)
        body["timestamp"] = ts
        body["sign"] = sign
    r = requests.post(FEISHU_WEBHOOK, json=body, timeout=30)
    log(f"飞书推送状态 {r.status_code} {r.text[:200]}")

def is_trade_day():
    try:
        cal = ak.tool_trade_date_hist_sina()
        dates = set(pd.to_datetime(cal["trade_date"]).dt.strftime("%Y%m%d"))
        return TODAY_STR in dates
    except Exception as e:
        log(f"[warn] 交易日历获取失败，继续执行: {e}")
        return True

def mkt(code):
    if code.startswith(("60", "68")): return "sh"
    if code.startswith(("00", "30")): return "sz"
    if code.startswith(("8", "4", "92")): return "bj"
    return "sz"

def f(x):
    try:
        v = float(x); return v if pd.notna(v) else None
    except Exception:
        return None

def enrich(code, name, snap, zt_map):
    out = {"代码": code, "简称": name, "收盘价": snap.get("最新价"),
           "涨幅": snap.get("涨跌幅"), "换手率": snap.get("换手率"),
           "量比": snap.get("量比"), "流通市值": snap.get("流通市值")}
    try:
        k = ak.stock_zh_a_hist(symbol=code, period="daily",
                               start_date=START_KLINE, end_date=TODAY_STR, adjust="qfq")
        if k is None or len(k) < 70:
            out["日K"] = "待核验(数据不足)"; return out
        k = k.tail(120).reset_index(drop=True)
        close = k["收盘"].astype(float); high = k["最高"].astype(float)
        low = k["最低"].astype(float); pct = k["涨跌幅"].astype(float)
        vol = k["成交量"].astype(float)

        is_gem = code.startswith(("30", "68"))
        thr = 14.0 if is_gem else 7.0
        last30 = pct.tail(30)
        out["30日最大单日涨幅"] = round(float(last30.max()), 2)
        out["30日大阳达标"] = bool((last30 >= thr).any())

        ma60 = close.rolling(60).mean()
        if len(ma60.dropna()) >= 11:
            d = ma60.diff().tail(10)
            out["60日线连续10天向上"] = bool((d > 0).all())
            out["站上60日线"] = bool(float(close.iloc[-1]) > float(ma60.iloc[-1]))
        else:
            out["60日线连续10天向上"] = "待核验"; out["站上60日线"] = "待核验"

        prev20_high = float(high.iloc[-21:-1].max()) if len(high) >= 21 else None
        out["突破20日高点"] = bool(prev20_high is not None and float(close.iloc[-1]) > prev20_high)
        out["非一字板"] = bool(float(high.iloc[-1]) > float(low.iloc[-1]))

        if len(high) >= 20:
            amp20 = float(high.tail(20).max() / low.tail(20).min() - 1)
            out["近20日振幅%"] = round(amp20 * 100, 2)
            out["横盘达标"] = bool(amp20 < 0.20)

        if len(vol) >= 25:
            pre5 = float(vol.iloc[-6:-1].mean())
            pre20 = float(vol.iloc[-26:-6].mean())
            if pre20 > 0:
                out["起爆前5日量能比"] = round(pre5 / pre20, 3)
                out["量能萎缩达标"] = bool(pre5 / pre20 < 0.8)

        mas = [close.rolling(n).mean().iloc[-1] for n in (5, 10, 20, 60)]
        if all(pd.notna(m) for m in mas):
            out["均线粘合度"] = round(float(max(mas) / min(mas) - 1) * 100, 2)
    except Exception as e:
        out["日K"] = f"待核验({type(e).__name__})"

    try:
        ff = ak.stock_individual_fund_flow(stock=code, market=mkt(code))
        if ff is not None and len(ff) > 0:
            out["主力净流入占比%"] = f(ff.iloc[-1].get("主力净流入-净占比"))
    except Exception as e:
        out["主力净流入占比%"] = f"待核验({type(e).__name__})"

    zt = zt_map.get(code)
    if zt is not None:
        seal = f(zt.get("封板资金")); fmv = f(zt.get("流通市值"))
        if seal is not None and fmv and fmv > 0:
            out["封单/流通市值%"] = round(seal / fmv * 100, 3)
    else:
        out["封单/流通市值%"] = "待核验(非涨停池)"

    out["距上方套牢区"] = "待核验"
    return out

def hard_filter(d):
    if d.get("30日大阳达标") is False: return False
    if d.get("60日线连续10天向上") is False: return False
    if d.get("站上60日线") is False: return False
    if d.get("突破20日高点") is False: return False
    if d.get("非一字板") is False: return False
    h = f(d.get("换手率")); mv = f(d.get("流通市值"))
    if h is not None and not (3 <= h <= 20): return False
    if mv is not None and mv >= 200e8: return False
    if "ST" in str(d.get("简称", "")): return False
    return True

def main():
    if not is_trade_day():
        log("非交易日，退出"); return
    if not LLM_API_KEY:
        log("缺少 DEEPSEEK_API_KEY"); sys.exit(1)

    log(f"模式：{'补跑' if RETRY_MODE else '首跑'}")
    log("拉取全市场快照...")
    df = ak.stock_zh_a_spot_em()
    for c in ["涨跌幅", "换手率", "量比", "流通市值", "最新价"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cand = df[(df["换手率"].between(3, 20)) & (df["流通市值"] < 200e8)
              & (~df["名称"].str.contains("ST", na=False))
              & (df["量比"] > 2.0)].copy()
    cand = cand.sort_values("量比", ascending=False).head(MAX_ENRICH)
    log(f"第一层过滤后候选 {len(cand)} 只（上限 {MAX_ENRICH}）")

    zt_map = {}
    try:
        zt = ak.stock_zt_pool_em(date=TODAY_STR)
        if zt is not None:
            zt_map = {str(r["代码"]): r for _, r in zt.iterrows()}
        log(f"涨停池 {len(zt_map)} 只")
    except Exception as e:
        log(f"[warn] 涨停池获取失败: {e}")

    enriched = []
    for i, (_, row) in enumerate(cand.iterrows(), 1):
        code = str(row["代码"]); name = str(row["名称"])
        log(f"[{i}/{len(cand)}] {code} {name}")
        try:
            d = enrich(code, name, row.to_dict(), zt_map)
        except Exception as e:
            log(f"  enrich 异常: {e}"); continue
        if hard_filter(d):
            enriched.append(d)
        time.sleep(0.15)

    log(f"硬过滤后 {len(enriched)} 只")
    if not enriched:
        report = "今日无达标标的。"
    else:
        sys_prompt = open("system_prompt.txt", encoding="utf-8").read()
        user_msg = f"""日期：{TODAY_STR}
候选股数据（JSON）：
{json.dumps(enriched, ensure_ascii=False, indent=1)}
请按 system_prompt 打分排序并输出 Markdown 报告。"""
        r = requests.post(LLM_URL,
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "temperature": 0.1,
                  "messages": [{"role": "system", "content": sys_prompt},
                               {"role": "user", "content": user_msg}]},
            timeout=300)
        r.raise_for_status()
        report = r.json()["choices"][0]["message"]["content"]

    log("===== 完整报告 =====")
    log(report)

    title = f"盘后量化大阳筛选报告 {TODAY_STR}"
    if RETRY_MODE:
        title += "（补跑）"
    body = report if len(report) <= 18000 else report[:18000] + "\n\n（完整版见 Actions 日志）"
    push_feishu(title, body)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
