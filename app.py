import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def npv(rate, cashflows):
    return sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows))

def irr_bisection(cashflows, low=-0.99, high=5.0, tol=1e-8, max_iter=400):
    f_low = npv(low, cashflows)
    f_high = npv(high, cashflows)
    if np.isnan(f_low) or np.isnan(f_high):
        return None
    if f_low * f_high > 0:
        return None
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0

def calc_payback_year(years, cumulative):
    for i in range(1, len(years)):
        prev_cum = cumulative[i-1]
        curr_cum = cumulative[i]
        if prev_cum < 0 and curr_cum >= 0:
            frac = (-prev_cum) / (curr_cum - prev_cum)
            return years[i-1] + frac
    return None

def award_rate(t, X, alpha0, Y):
    """
    Award rate alpha_t:
      - for t <= X: alpha0
      - for t > X:  alpha0 * (1 - Y)^(t - X)
    where t is year index starting at 1,
          X is years at initial award rate,
          alpha0 in [0,1] is initial award rate,
          Y in [0,1] is YoY decline rate.
    """
    if t <= X:
        return alpha0
    return alpha0 * ((1.0 - Y) ** (t - X))

def main():
    st.set_page_config(page_title="需給調整市場：最小IRR/NPV計算", layout="wide")
    st.title("需給調整市場：最小IRR/NPV計算（落札率α0＋低下Y＋β＋γ＋手数料＋O&M＋廃止費用）")

    with st.sidebar:
        st.header("入力")
        price = st.number_input("単価 (円/(kW・コマ))", min_value=0.0, value=5.0, step=0.1)
        slots_per_day = st.number_input("一日に参加するコマ数", min_value=0, value=48, step=1)
        days_per_year = st.number_input("年間で参加する日数", min_value=0, value=353, step=1)
        power_kw = st.number_input("ESSの出力 (kW)", min_value=0.0, value=2000.0, step=100.0)
        energy_kwh = st.number_input("ESSの容量 (kWh)", min_value=0.0, value=7000.0, step=100.0)
        unit_cost = st.number_input("システム単価 (円/kWh)", min_value=0.0, value=60000.0, step=1000.0)
        years = int(st.number_input("評価年数 (年)", min_value=1, value=15, step=1))

        st.divider()
        st.subheader("係数（約定/運用）")
        beta_percent = st.number_input("β：約定量ディレーティング係数（%）", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        gamma_percent = st.number_input("γ：パフォーマンス・運用制約係数（%）", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        beta = beta_percent / 100.0
        gamma = gamma_percent / 100.0

        st.divider()
        st.subheader("落札率モデル（修正版）")
        alpha0_percent = st.number_input("最初のX年は落札率 α%（α）", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        X = int(st.number_input("最初のX年（年）", min_value=0, value=3, step=1))
        Y_percent = st.number_input("X年以降、前年比でY%ずつ低下 (Y%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        alpha0 = alpha0_percent / 100.0
        Y = Y_percent / 100.0

        st.divider()
        fee_rate = st.number_input("RA+AC 手数料率 (%)", min_value=0.0, value=10.0, step=0.5)
        om_cost_per_kw_year = st.number_input("O&M費 (円/kW/年)", min_value=0.0, value=3000.0, step=100.0)
        decommission_rate = st.number_input("廃止措置費用率（CAPEX比, %）", min_value=0.0, value=5.0, step=0.5)
        discount_rate = st.number_input("割引率 r", min_value=0.0, value=0.05, step=0.01, format="%.4f")

    capex = unit_cost * energy_kwh

    # Apply beta to kW and gamma to annual participation days
    effective_power_kw = power_kw * beta
    effective_days = days_per_year * gamma

    base_gross_revenue = price * slots_per_day * effective_days * effective_power_kw

    om_year = om_cost_per_kw_year * power_kw  # per requirement: based on nameplate output
    decommission_cost = capex * decommission_rate / 100.0

    years_list = [0]
    award_list = [np.nan]
    gross_list = [0.0]
    fee_list = [0.0]
    om_list = [0.0]
    decom_list = [0.0]
    cf_list = [-capex]
    cum_list = [-capex]

    cum = -capex

    for t in range(1, years + 1):
        alpha_t = award_rate(t, X, alpha0, Y)
        gross = base_gross_revenue * alpha_t
        fee = gross * fee_rate / 100.0
        net = gross - fee - om_year
        decom = 0.0
        if t == years:
            decom = decommission_cost
            net = net - decom

        cum += net

        years_list.append(t)
        award_list.append(alpha_t)
        gross_list.append(gross)
        fee_list.append(fee)
        om_list.append(om_year)
        decom_list.append(decom)
        cf_list.append(net)
        cum_list.append(cum)

    df = pd.DataFrame({
        "Year": years_list,
        "AwardRate": award_list,
        "GrossRevenue": gross_list,
        "Fee": fee_list,
        "OM": om_list,
        "DecommissionCost": decom_list,
        "CashFlow": cf_list,
        "CumulativeCashFlow": cum_list,
    })

    irr = irr_bisection(cf_list)
    npv_val = npv(discount_rate, cf_list)
    payback_year = calc_payback_year(years_list, cum_list)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("結果（KPI）")
        st.metric("CAPEX (円)", f"{capex:,.0f}")
        st.metric("有効出力 (kW) = 出力×β", f"{effective_power_kw:,.2f}")
        st.metric("有効参加日数 (日/年) = 日数×γ", f"{effective_days:,.2f}")
        st.metric("ベース年間総収入（落札率=100%時）(円/年)", f"{base_gross_revenue:,.0f}")
        st.metric("O&M費 (円/年)", f"{om_year:,.0f}")
        st.metric("廃止措置費用（最終年）", f"{decommission_cost:,.0f}")

        if payback_year is not None:
            st.metric("回収完了年度", f"{payback_year:.2f} 年")
        else:
            st.warning("評価期間内に回収できません")

        st.metric(f"NPV (r={discount_rate:.2%})", f"{npv_val:,.0f}")
        if irr is None:
            st.warning("IRRが計算できません（NPVの符号が変わらない可能性）。")
        else:
            st.metric("IRR", f"{irr:.2%}")

        st.caption("※落札率は「最初X年=α%、以降は前年比Y%低下」。βは出力、γは参加日数に乗算。")

    with right:
        st.subheader("年次キャッシュフロー（落札率＋β＋γ込み）")
        st.dataframe(df, use_container_width=True)

    st.subheader("キャッシュフロー（年次）と累積キャッシュフロー")
    fig, ax1 = plt.subplots()
    ax1.bar(df["Year"], df["CashFlow"])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cash Flow (JPY)")
    ax2 = ax1.twinx()
    ax2.plot(df["Year"], df["CumulativeCashFlow"])
    ax2.set_ylabel("Cumulative Cash Flow (JPY)")
    st.pyplot(fig)

    st.subheader("落札率の推移（参考）")
    fig2, ax = plt.subplots()
    ax.plot(df["Year"], df["AwardRate"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Award Rate")
    ax.set_ylim(0, 1.05)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
