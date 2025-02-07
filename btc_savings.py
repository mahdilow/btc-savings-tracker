import streamlit as st
import pandas as pd
import requests
import datetime
import jdatetime
import plotly.express as px

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------


@st.cache_data(ttl=3600)
def get_historical_btc_prices(from_date: datetime.date, to_date: datetime.date) -> dict:
    """
    Fetch historical BTC prices (in USD) from CoinGecko between from_date and to_date.
    Returns a dictionary mapping datetime.date to price (using the last price available in that day).
    """
    from_ts = int(datetime.datetime.combine(from_date, datetime.time()).timestamp())
    to_ts = int(datetime.datetime.combine(to_date, datetime.time()).timestamp())
    url = (
        f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={from_ts}&to={to_ts}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        prices = data.get("prices", [])
        price_by_date = {}
        for ts, price in prices:
            dt = datetime.datetime.fromtimestamp(ts / 1000).date()
            # Overwrite so the last price for that day remains.
            price_by_date[dt] = price
        return price_by_date
    else:
        st.error("Error fetching historical BTC data.")
        return {}


def get_current_btc_price() -> float:
    """
    Fetch the current BTC price in USD using CoinGecko's API.
    Returns a float if successful; otherwise, None.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["bitcoin"]["usd"]
        else:
            return None
    except Exception as e:
        st.error("Error fetching current BTC price.")
        return None


# -------------------------------------------------------
# Main App
# -------------------------------------------------------


def main():
    st.title("Bitcoin Savings & Profit Tracker (Jalali Version)")
    st.write(
        "Track your BTC buys, see your overall investment, and analyze your portfolio performance over time."
    )

    # Initialize session state for transactions if needed.
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = []

    # Use tabs to separate Summary from Timeline Analysis.
    tabs = st.tabs(["Summary", "Timeline Analysis"])

    # ======================================================
    # TAB 1: SUMMARY (Entry, Listing, and Basic Analysis)
    # ======================================================
    with tabs[0]:
        st.header("Transaction Entry and Summary")
        st.markdown("### Add a New Transaction")

        # --------------------------
        # Jalali Date Picker
        # --------------------------
        st.info("Select the purchase date using the Jalali (Persian) calendar.")
        today_jalali = jdatetime.date.today()
        # Year: allow a range (e.g. 1370 to current Jalali year)
        years = list(range(1370, today_jalali.year + 1))
        default_year_index = (
            years.index(today_jalali.year)
            if today_jalali.year in years
            else len(years) - 1
        )
        selected_year = st.selectbox("Year (Jalali)", years, index=default_year_index)
        # Month: 1 to 12
        selected_month = st.selectbox(
            "Month (Jalali)", list(range(1, 13)), index=today_jalali.month - 1
        )
        # Compute maximum day for the given month and year in the Jalali calendar.
        if selected_month <= 6:
            max_day = 31
        elif selected_month <= 11:
            max_day = 30
        else:  # month 12
            max_day = 30 if jdatetime.isleap(selected_year) else 29
        days = list(range(1, max_day + 1))
        default_day_index = today_jalali.day - 1 if today_jalali.day <= max_day else 0
        selected_day = st.selectbox("Day (Jalali)", days, index=default_day_index)

        try:
            jalali_date = jdatetime.date(selected_year, selected_month, selected_day)
            # Convert to Gregorian for internal processing.
            gregorian_date = jalali_date.togregorian()
        except Exception as e:
            st.error("Invalid Jalali date selected.")
            return

        # --------------------------
        # Other Transaction Inputs
        # --------------------------
        amount = st.number_input(
            "BTC Amount", min_value=0.0, format="%.8f", step=0.00000001
        )
        price = st.number_input(
            "Price per BTC (USD)", min_value=0.0, format="%.2f", step=0.01
        )

        if st.button("Add Transaction"):
            if amount > 0 and price > 0:
                txn = {
                    "jalali_date": jalali_date.strftime("%Y-%m-%d"),
                    "gregorian_date": gregorian_date.isoformat(),  # e.g., "2025-02-07"
                    "amount": amount,
                    "price": price,
                    "total_cost": amount * price,
                }
                st.session_state["transactions"].append(txn)
                st.success("Transaction added!")
            else:
                st.error("Amount and Price must be greater than zero.")

        st.markdown("### Your Transactions")
        if st.session_state["transactions"]:
            # Show each transaction with a delete option.
            for idx, txn in enumerate(st.session_state["transactions"]):
                cols = st.columns([1, 1, 1, 1, 1])
                cols[0].write(f"**Date:** {txn['jalali_date']}")
                cols[1].write(f"**BTC:** {txn['amount']:.8f}")
                cols[2].write(f"**Price:** ${txn['price']:,.2f}")
                cols[3].write(f"**Cost:** ${txn['total_cost']:,.2f}")
                if cols[4].button("Delete", key=f"delete_{idx}"):
                    st.session_state["transactions"].pop(idx)
                    st.experimental_rerun()

            # Also display a table of transactions.
            df = pd.DataFrame(st.session_state["transactions"])
            df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
            df = df.sort_values(by="greg_date")
            st.dataframe(df[["jalali_date", "amount", "price", "total_cost"]])

            # Basic Summary
            total_btc = df["amount"].sum()
            total_cost = df["total_cost"].sum()
            st.write(f"**Total BTC Purchased:** {total_btc:.8f}")
            st.write(f"**Total Invested:** ${total_cost:,.2f}")

            current_btc_price = get_current_btc_price()
            if current_btc_price is not None:
                st.write(f"**Current BTC Price:** ${current_btc_price:,.2f}")
                current_value = total_btc * current_btc_price
                profit = current_value - total_cost
                profit_percent = (profit / total_cost * 100) if total_cost > 0 else 0
                st.write(f"**Current Value of Holdings:** ${current_value:,.2f}")
                st.write(f"**Profit / Loss:** ${profit:,.2f} ({profit_percent:.2f}%)")

                # Simple bar chart: Invested vs Current Value
                chart_df = pd.DataFrame(
                    {
                        "Metric": ["Invested", "Current Value"],
                        "USD": [total_cost, current_value],
                    }
                )
                st.bar_chart(chart_df.set_index("Metric"))
            else:
                st.error("Unable to fetch current BTC price.")
        else:
            st.info("No transactions added yet.")

        if st.button("Clear All Transactions"):
            st.session_state["transactions"] = []
            st.experimental_rerun()

    # ======================================================
    # TAB 2: TIMELINE ANALYSIS (Advanced Charts)
    # ======================================================
    with tabs[1]:
        st.header("Timeline Analysis")
        if st.session_state["transactions"]:
            # Convert transactions to a DataFrame and sort by Gregorian date.
            df = pd.DataFrame(st.session_state["transactions"])
            df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
            df = df.sort_values(by="greg_date")
            earliest_date = df["greg_date"].min()
            today = datetime.date.today()

            # Fetch historical BTC prices between earliest transaction and today.
            price_by_date = get_historical_btc_prices(earliest_date, today)

            # Build a daily timeline:
            timeline = []
            cumulative_btc = 0.0
            cumulative_cost = 0.0
            txn_idx = 0
            txns = df.to_dict("records")
            num_txns = len(txns)
            current_day = earliest_date

            while current_day <= today:
                # Add any transactions that occurred on the current day.
                while txn_idx < num_txns and txns[txn_idx]["greg_date"] == current_day:
                    cumulative_btc += txns[txn_idx]["amount"]
                    cumulative_cost += txns[txn_idx]["total_cost"]
                    txn_idx += 1
                btc_price = price_by_date.get(current_day)
                if btc_price is None:
                    # If no price for the day, use the previous day's price (or current price if first day).
                    btc_price = (
                        timeline[-1]["btc_price"]
                        if timeline
                        else get_current_btc_price()
                    )
                portfolio_value = cumulative_btc * btc_price
                profit = portfolio_value - cumulative_cost
                roi = (
                    (portfolio_value / cumulative_cost * 100 - 100)
                    if cumulative_cost > 0
                    else 0
                )
                timeline.append(
                    {
                        "date": current_day,
                        "cumulative_btc": cumulative_btc,
                        "cumulative_cost": cumulative_cost,
                        "btc_price": btc_price,
                        "portfolio_value": portfolio_value,
                        "profit": profit,
                        "roi": roi,
                    }
                )
                current_day += datetime.timedelta(days=1)

            timeline_df = pd.DataFrame(timeline)

            st.subheader("Cumulative Investment vs Portfolio Value")
            fig1 = px.line(
                timeline_df,
                x="date",
                y=["cumulative_cost", "portfolio_value"],
                labels={"value": "USD", "variable": "Metric"},
                title="Cumulative Investment vs Portfolio Value Over Time",
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Profit Over Time")
            fig2 = px.line(
                timeline_df,
                x="date",
                y="profit",
                title="Profit Over Time (USD)",
                labels={"profit": "Profit (USD)"},
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Return on Investment (ROI) Over Time")
            fig3 = px.line(
                timeline_df,
                x="date",
                y="roi",
                title="ROI Over Time (%)",
                labels={"roi": "ROI (%)"},
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Add transactions in the Summary tab to see timeline analysis.")


if __name__ == "__main__":
    main()
