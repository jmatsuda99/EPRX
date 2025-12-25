
import streamlit as st
import numpy as np
import pandas as pd
import json
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
                    total = int(json.load(f).get("total", 0))
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
            return int(json.load(f).get("total", 0))
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
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < 1e-8:
            return mid
        if npv(low, cashflows) * f_mid <= 0:
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

# -----------------------------
# Scenario models
# -----------------------------
def award_rate_decline(t: int, X: int, alpha0: float, Y: float) -> float:
    """Model A: award rate alpha declines after X years."""
    return alpha0 if t <= X else alpha0 * ((1 - Y) ** (t - X))

def price_decline(t: int, X: int, price1: float, Y: float) -> float:
    """Model B: price declines after X years."""
    return price1 if t <= X else price1 * ((1 - Y) ** (t - X))

# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title="需給調整市場IRR/NPV", layout="wide")
    increment_counter_once()

    st.title("需給調整市場：IRR / NPV 計算アプリ（v17）")
    st.caption(f"Access count: {read_counter()}")

    with st.sidebar:
        st.header("収入モデルの選択")
        revenue_model = st.radio(
            "落札率/単価の扱い",
            [
                "モデルA：落札率が年次で低下（単価は一定）",
                "モデルB：落札率は一定（単価が年次で低下）",
            ],
        )

        st.divider()
        st.header("CAPEX入力方式")
        capex_mode = st.radio(
            "CAPEXの算定方法",
            ["単価方式（円/kWh × 容量）", "積算方式（機器費＋工事費）"]
        )

        st.divider()
        st.header("市場・運用条件")
        price = st.number_input("単価 (円/(kW・コマ)) ※モデルBでは初年度単価α1", value=5.0, min_value=0.0)
        slots = st.number_input("コマ数/日", value=48, min_value=0)
        days = st.number_input("参加日数/年", value=353, min_value=0)
        power = st.number_input("出力 (kW)", value=2000.0, min_value=0.0)
        years = int(st.number_input("評価年数 (年)", value=15, min_value=1))

        st.divider()
        st.header("係数・率（％入力）")
        beta_pct = st.number_input("β 約定量ディレーティング係数（％）", value=100.0, min_value=0.0, max_value=200.0)
        gamma_pct = st.number_input("γ パフォーマンス・運用制約係数（％）", value=100.0, min_value=0.0, max_value=200.0)
        beta = beta_pct / 100.0
        gamma = gamma_pct / 100.0

        X = int(st.number_input("X：維持年数 (年)", value=3, min_value=0))
        Y_pct = st.number_input("Y：X年以降の低下率（％/年）", value=10.0, min_value=0.0, max_value=100.0)
        Y = Y_pct / 100.0

        if revenue_model.startswith("モデルA"):
            alpha0_pct = st.number_input("初期落札率 α（％）", value=100.0, min_value=0.0, max_value=100.0)
            alpha0 = alpha0_pct / 100.0
            fixed_award = None
        else:
            fixed_award_pct = st.number_input("落札率（一定）（％）", value=100.0, min_value=0.0, max_value=100.0)
            fixed_award = fixed_award_pct / 100.0
            alpha0 = None

        st.divider()
        st.header("コスト・評価条件（％入力）")
        fee_pct = st.number_input("RA+AC 手数料率（％）", value=10.0, min_value=0.0, max_value=100.0)
        fee = fee_pct / 100.0
        om_kw = st.number_input("O&M費 (円/kW/年)", value=3000.0, min_value=0.0)
        decom_pct = st.number_input("廃止費率（CAPEX比）（％）", value=5.0, min_value=0.0, max_value=100.0)
        decom = decom_pct / 100.0
        r_pct = st.number_input("割引率 r（％）", value=5.0, min_value=0.0, max_value=100.0)
        r = r_pct / 100.0

        st.divider()
        st.header("CAPEX入力")
        if capex_mode.startswith("単価方式"):
            unit_cost = st.number_input("システム単価 (円/kWh)", value=60000.0, min_value=0.0)
            energy = st.number_input("ESS容量 (kWh)", value=7000.0, min_value=0.0)
            capex = unit_cost * energy
            capex_note = "単価方式（円/kWh × 容量）"
        else:
            equipment_cost = st.number_input("機器費 (円)", value=300_000_000.0, step=1_000_000.0, min_value=0.0)
            construction_cost = st.number_input("工事費 (円)", value=100_000_000.0, step=1_000_000.0, min_value=0.0)
            capex = equipment_cost + construction_cost
            capex_note = "積算方式（機器費＋工事費）"

    # -----------------------------
    # Derived inputs (V12 feature preserved)
    # -----------------------------
    effective_power = power * beta
    effective_days = days * gamma
    om_year = om_kw * power
    decom_cost = capex * decom

    base_revenue_coeff = slots * effective_days * effective_power  # multiplier for price & award
    base_revenue_at_price = price * base_revenue_coeff  # for display reference

    derived_df = pd.DataFrame({
        "項目": [
            "CAPEX",
            "CAPEX算定方式",
            "有効出力（出力×β）",
            "有効参加日数（日数×γ）",
            "ベース年間総収入（単価×…、モデルAの基準）",
            "年間O&M費（O&M×出力）",
            "廃止措置費用（最終年, CAPEX×率）",
            "収入モデル",
        ],
        "値": [
            f"{capex:,.0f}",
            capex_note,
            f"{effective_power:,.2f}",
            f"{effective_days:,.2f}",
            f"{base_revenue_at_price:,.0f}",
            f"{om_year:,.0f}",
            f"{decom_cost:,.0f}",
            revenue_model,
        ],
        "単位": ["円", "-", "kW", "日/年", "円/年", "円/年", "円", "-"]
    })

    st.subheader("📌 計算により導出された入力値一覧")
    st.dataframe(derived_df, use_container_width=True)

    # -----------------------------
    # Cashflow & annual table
    # -----------------------------
    years_list = [0]
    award_list = [np.nan]
    price_list = [np.nan]
    gross_list = [0.0]
    fee_list = [0.0]
    om_list = [0.0]
    decom_list = [0.0]
    cf = [-capex]
    cum = [-capex]

    for t in range(1, years + 1):
        if revenue_model.startswith("モデルA"):
            a_t = award_rate_decline(t, X, alpha0, Y)
            p_t = price
        else:
            a_t = fixed_award
            p_t = price_decline(t, X, price, Y)

        gross = p_t * base_revenue_coeff * a_t
        fee_y = gross * fee
        net = gross - fee_y - om_year

        decom_y = 0.0
        if t == years:
            decom_y = decom_cost
            net -= decom_y

        years_list.append(t)
        award_list.append(a_t)
        price_list.append(p_t)
        gross_list.append(gross)
        fee_list.append(fee_y)
        om_list.append(om_year)
        decom_list.append(decom_y)

        cf.append(net)
        cum.append(cum[-1] + net)

    irr = irr_bisection(cf)
    npv_val = npv(r, cf)
    payback = calc_payback_year(cum)

    st.subheader("📊 結果指標")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAPEX", f"{capex:,.0f} 円")
    c2.metric("NPV", f"{npv_val:,.0f} 円")
    c3.metric("IRR", f"{irr:.2%}" if irr is not None else "計算不可")
    c4.metric("回収年", f"{payback:.2f} 年" if payback is not None else "回収不可")

    df_cf = pd.DataFrame({
        "Year": years_list,
        "UnitPrice": price_list,
        "AwardRate": award_list,
        "GrossRevenue": gross_list,
        "Fee": fee_list,
        "OM": om_list,
        "Decommission": decom_list,
        "CashFlow": cf,
        "CumulativeCashFlow": cum,
    })

    st.subheader("年次キャッシュフロー（年次）と累積キャッシュフロー")
    st.dataframe(df_cf, use_container_width=True)

    # Combined chart: annual CF (bar) + cumulative CF (line)
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.bar(df_cf["Year"], df_cf["CashFlow"])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cash Flow (JPY)")
    ax2 = ax1.twinx()
    ax2.plot(df_cf["Year"], df_cf["CumulativeCashFlow"])
    ax2.set_ylabel("Cumulative Cash Flow (JPY)")
    st.pyplot(fig)

    # Optional: trajectories
    st.subheader("参考：単価と落札率の推移")
    fig2, axp = plt.subplots()
    axp.plot(df_cf["Year"], df_cf["UnitPrice"])
    axp.set_xlabel("Year")
    axp.set_ylabel("Unit Price")
    st.pyplot(fig2)

    fig3, axa = plt.subplots()
    axa.plot(df_cf["Year"], df_cf["AwardRate"])
    axa.set_xlabel("Year")
    axa.set_ylabel("Award Rate")
    axa.set_ylim(0, 1.05)
    st.pyplot(fig3)

if __name__ == "__main__":
    main()
