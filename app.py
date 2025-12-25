
import streamlit as st
import numpy as np
import pandas as pd
import json, os
from contextlib import contextmanager

# -----------------------------
# Access counter
# -----------------------------
COUNTER_FILE = "access_counter.json"

@contextmanager
def file_lock(fp):
    try:
        import fcntl
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    except Exception:
        yield

def increment_counter_once():
    if st.session_state.get("_counted", False):
        return
    total = 0
    try:
        with open(COUNTER_FILE, "a+", encoding="utf-8") as f:
            with file_lock(f):
                f.seek(0)
                try:
                    total = json.load(f).get("total", 0)
                except Exception:
                    total = 0
                total += 1
                f.seek(0)
                f.truncate()
                json.dump({"total": total}, f)
    except Exception:
        pass
    st.session_state["_counted"] = True

def read_counter():
    try:
        with open(COUNTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("total", 0)
    except Exception:
        return 0

# -----------------------------
# Finance helpers
# -----------------------------
def npv(rate, cashflows):
    return sum(cf / ((1+rate)**t) for t, cf in enumerate(cashflows))

def irr_bisection(cashflows):
    low, high = -0.99, 5.0
    if npv(low, cashflows) * npv(high, cashflows) > 0:
        return None
    for _ in range(300):
        mid = (low + high) / 2
        if abs(npv(mid, cashflows)) < 1e-8:
            return mid
        if npv(low, cashflows) * npv(mid, cashflows) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2

def calc_payback_year(cum):
    for i in range(1, len(cum)):
        if cum[i-1] < 0 <= cum[i]:
            frac = (-cum[i-1]) / (cum[i] - cum[i-1])
            return (i-1) + frac
    return None

def award_rate(t, X, alpha0, Y):
    return alpha0 if t <= X else alpha0 * ((1-Y)**(t-X))

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title="需給調整市場IRR/NPV", layout="wide")
    increment_counter_once()

    st.title("需給調整市場：IRR / NPV 計算アプリ（v14）")
    st.caption(f"Access count: {read_counter()}")

    # =============================
    # Sidebar inputs
    # =============================
    with st.sidebar:
        st.header("CAPEX入力方式")
        capex_mode = st.radio(
            "CAPEXの算定方法",
            ["単価方式（円/kWh × 容量）", "積算方式（機器費＋工事費）"]
        )

        st.divider()
        st.header("市場・運用条件")
        price = st.number_input("単価 (円/(kW・コマ))", value=5.0)
        slots = st.number_input("コマ数/日", value=48)
        days = st.number_input("参加日数/年", value=353)
        power = st.number_input("出力 (kW)", value=2000.0)
        years = int(st.number_input("評価年数 (年)", value=15))

        beta = st.number_input("β 約定量係数", value=1.0)
        gamma = st.number_input("γ 稼働率係数", value=1.0)

        st.divider()
        st.header("落札率モデル")
        alpha0 = st.number_input("初期落札率 α", value=1.0)
        X = int(st.number_input("α維持年数 X", value=3))
        Y = st.number_input("低下率 Y", value=0.1)

        st.divider()
        st.header("コスト・評価条件")
        fee = st.number_input("RA+AC 手数料率", value=0.1)
        om_kw = st.number_input("O&M費 (円/kW/年)", value=3000.0)
        decom = st.number_input("廃止費率 (CAPEX比)", value=0.05)
        r = st.number_input("割引率 r", value=0.05)

        st.divider()
        st.header("CAPEX入力")
        if capex_mode.startswith("単価方式"):
            unit_cost = st.number_input("システム単価 (円/kWh)", value=60000.0)
            energy = st.number_input("ESS容量 (kWh)", value=7000.0)
            capex = unit_cost * energy
            capex_note = "単価方式（円/kWh × 容量）"
        else:
            equipment_cost = st.number_input("機器費 (円)", value=300_000_000.0, step=1_000_000.0)
            construction_cost = st.number_input("工事費 (円)", value=100_000_000.0, step=1_000_000.0)
            capex = equipment_cost + construction_cost
            capex_note = "積算方式（機器費＋工事費）"

    # =============================
    # Derived inputs (V12 feature)
    # =============================
    effective_power = power * beta
    effective_days = days * gamma
    base_revenue = price * slots * effective_days * effective_power
    om_year = om_kw * power
    decom_cost = capex * decom

    derived_df = pd.DataFrame({
        "項目": [
            "CAPEX",
            "CAPEX算定方式",
            "有効出力",
            "有効参加日数",
            "ベース年間総収入",
            "年間O&M費",
            "廃止措置費用"
        ],
        "値": [
            f"{capex:,.0f}",
            capex_note,
            f"{effective_power:,.2f}",
            f"{effective_days:,.2f}",
            f"{base_revenue:,.0f}",
            f"{om_year:,.0f}",
            f"{decom_cost:,.0f}"
        ],
        "単位": [
            "円", "-", "kW", "日/年", "円/年", "円/年", "円"
        ]
    })

    st.subheader("📌 計算により導出された入力値一覧")
    st.dataframe(derived_df, use_container_width=True)

    # =============================
    # Cashflow
    # =============================
    cf = [-capex]
    cum = [-capex]

    for t in range(1, years+1):
        revenue = base_revenue * award_rate(t, X, alpha0, Y)
        net = revenue * (1-fee) - om_year
        if t == years:
            net -= decom_cost
        cf.append(net)
        cum.append(cum[-1] + net)

    irr = irr_bisection(cf)
    npv_val = npv(r, cf)
    payback = calc_payback_year(cum)

    # =============================
    # Results
    # =============================
    st.subheader("📊 結果指標")
    st.metric("CAPEX", f"{capex:,.0f} 円")
    st.metric("NPV", f"{npv_val:,.0f} 円")
    st.metric("IRR", f"{irr:.2%}" if irr else "計算不可")
    if payback:
        st.metric("回収年", f"{payback:.2f} 年")
    else:
        st.warning("評価期間内に回収不可")

    st.subheader("年次キャッシュフローと累積キャッシュフロー")
    df_cf = pd.DataFrame({
        "Year": list(range(0, years+1)),
        "CashFlow": cf,
        "Cumulative": cum
    })
    st.dataframe(df_cf, use_container_width=True)

if __name__ == "__main__":
    main()
