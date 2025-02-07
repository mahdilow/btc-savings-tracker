import streamlit as st
import pandas as pd
import requests
import datetime
import jdatetime
import plotly.express as px
import io
import xlsxwriter
from typing import Dict, List
from PIL import Image
import plotly.graph_objects as go
from io import BytesIO
import time

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------


@st.cache_data(ttl=3600)
def get_historical_btc_prices(from_date: datetime.date, to_date: datetime.date) -> dict:
    """
    Fetch historical BTC prices (in USD) from CoinGecko between from_date and to_date.
    Returns a dictionary mapping datetime.date to price.
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
            # Use the last available price for each day.
            price_by_date[dt] = price
        return price_by_date
    else:
        st.error("Error fetching historical BTC data.")
        return {}


def get_current_btc_price(max_retries: int = 3, retry_delay: float = 1.0) -> float:
    """
    Fetch the current BTC price in USD using CoinGecko's API.
    Includes automatic retries and returns None if all retries fail.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data["bitcoin"]["usd"]
            else:
                print(
                    f"Attempt {attempt + 1} failed with status code: {response.status_code}"
                )
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            time.sleep(retry_delay)

    return None


def calculate_advanced_metrics(df: pd.DataFrame, current_btc_price: float) -> Dict:
    """
    Calculate advanced investment metrics from transaction history
    """
    metrics = {
        "total_invested": df["total_cost"].sum(),
        "total_btc": df["amount"].sum(),
        "average_buy_price": df["total_cost"].sum() / df["amount"].sum(),
        "highest_buy": df["price"].max(),
        "lowest_buy": df["price"].min(),
        "price_volatility": df["price"].std(),
        "current_value": df["amount"].sum() * current_btc_price,
    }

    metrics["total_profit"] = metrics["current_value"] - metrics["total_invested"]
    metrics["roi_percent"] = (
        (metrics["total_profit"] / metrics["total_invested"] * 100)
        if metrics["total_invested"] > 0
        else 0
    )

    # Monthly analysis
    df["month"] = pd.to_datetime(df["gregorian_date"]).dt.to_period("M")
    monthly_stats = (
        df.groupby("month")
        .agg({"amount": "sum", "total_cost": "sum", "price": "mean"})
        .reset_index()
    )

    metrics["best_month"] = monthly_stats.loc[
        monthly_stats["amount"].idxmax(), "month"
    ].strftime("%Y-%m")
    metrics["highest_monthly_investment"] = monthly_stats["total_cost"].max()

    return metrics


def export_to_excel(transactions: List[Dict]) -> bytes:
    """
    Convert transactions data to Excel file
    """
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet("Transactions")

    # Add headers
    headers = [
        "Jalali Date",
        "Gregorian Date",
        "BTC Amount",
        "Price (USD)",
        "Total Cost (USD)",
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    # Add data
    for row, txn in enumerate(transactions, start=1):
        worksheet.write(row, 0, txn["jalali_date"])
        worksheet.write(row, 1, txn["gregorian_date"])
        worksheet.write(row, 2, txn["amount"])
        worksheet.write(row, 3, txn["price"])
        worksheet.write(row, 4, txn["total_cost"])

    workbook.close()
    return output.getvalue()


def create_summary_image(
    metrics: Dict, monthly_chart: pd.DataFrame, current_price: float
) -> bytes:
    """
    Create a beautiful summary image with key metrics and charts
    """
    # Create a figure with subplots
    fig = go.Figure()

    # Add monthly investment chart
    fig.add_trace(
        go.Bar(
            x=monthly_chart["month"],
            y=monthly_chart["total_cost"],
            name="Monthly Investment",
            text=[f"${x:,.0f}" for x in monthly_chart["total_cost"]],
            textposition="outside",
        )
    )

    # Add key metrics as annotations
    fig.add_annotation(
        text=(
            f"Investment Summary\n"
            f"Total Invested: ${metrics['total_invested']:,.2f}\n"
            f"Current Value: ${metrics['current_value']:,.2f}\n"
            f"Total ROI: {metrics['roi_percent']:.2f}%\n"
            f"Average Buy Price: ${metrics['average_buy_price']:,.2f}\n"
            f"Current BTC Price: ${current_price:,.2f}\n"
            f"Total BTC: {metrics['total_btc']:.8f}"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=14),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
    )

    # Update layout
    fig.update_layout(
        title="Bitcoin Investment Summary",
        template="plotly_white",
        height=800,
        showlegend=False,
        xaxis_title="Month",
        yaxis_title="Investment (USD)",
    )

    # Convert to image
    img_bytes = fig.to_image(format="png", scale=2)
    return img_bytes


def is_jalali_leap_year(year: int) -> bool:
    """
    Determine if a given year in the Jalali calendar is a leap year.
    """
    cycle_year = year % 33
    return cycle_year in [1, 5, 9, 13, 17, 22, 26, 30]


def initialize_btc_price():
    """
    Initialize BTC price in session state if not already present
    """
    if "current_btc_price" not in st.session_state:
        price = get_current_btc_price()
        st.session_state["current_btc_price"] = price
        st.session_state["price_fetch_time"] = datetime.datetime.now().strftime(
            "%H:%M:%S"
        )


def display_price_section(location, section_name: str):
    """
    Display the current BTC price section with retry button.
    Returns the current price or None if unavailable.
    """
    if st.session_state.get("current_btc_price") is not None:
        return st.session_state["current_btc_price"]
    else:
        col1, col2 = location.columns([3, 1])
        col1.error(
            "Unable to fetch current BTC price. Please check your internet connection."
        )
        if col2.button("🔄 Retry", key=f"retry_{section_name}"):
            price = get_current_btc_price()
            if price is not None:
                st.session_state["current_btc_price"] = price
                st.session_state["price_fetch_time"] = datetime.datetime.now().strftime(
                    "%H:%M:%S"
                )
                st.rerun()
        return None


# -------------------------------------------------------
# Main Application
# -------------------------------------------------------


def main():
    st.title("Bitcoin Savings & Profit Tracker")
    st.write(
        "Track your BTC purchases, analyze your overall investment, and view timeline analytics."
    )

    # Initialize BTC price at startup
    initialize_btc_price()

    # Add a small info text showing when price was fetched
    if "price_fetch_time" in st.session_state:
        st.caption(f"Price last fetched at: {st.session_state['price_fetch_time']}")

    # Initialize session state for transactions.
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = []

    # Create tabs for different sections
    tabs = st.tabs(
        [
            "Summary",
            "Advanced Analytics",
            "DCA Planning",
            "Timeline Analysis",
            "Export Data",
        ]
    )

    # ======================================================
    # TAB 1: SUMMARY (CSV Import, Manual Entry, & Basic Analysis)
    # ======================================================
    with tabs[0]:
        st.header("Transaction Entry and Summary")

        # -----------------------------------------------
        # CSV Import Section with Instructions
        # -----------------------------------------------
        with st.expander("Import CSV of Transactions"):
            st.write(
                "You can upload a CSV file containing your BTC purchase transactions. **How should your CSV look?**"
            )
            st.markdown(
                """
                - **Required Columns:** `jalali_date`, `amount`, and `price`
                - **jalali_date:** The purchase date in the Persian (Jalali) calendar, formatted as `YYYY-MM-DD` (e.g., `1402-01-15`).
                - **amount:** The amount of BTC purchased (decimal number).
                - **price:** The price per BTC in USD at which you bought.
                
                **Example CSV Content:**
                
                    jalali_date,amount,price
                    1400-12-16,0.00050000,95000
                    1403-01-11,0.00022091,62321    
                """
            )
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)
                    required_cols = ["jalali_date", "amount", "price"]
                    if all(col in csv_df.columns for col in required_cols):
                        if st.button("Import CSV Transactions"):
                            for idx, row in csv_df.iterrows():
                                j_date_str = str(row["jalali_date"])
                                try:
                                    # Expecting the format YYYY-MM-DD.
                                    year, month, day = map(int, j_date_str.split("-"))
                                    j_date = jdatetime.date(year, month, day)
                                    g_date = j_date.togregorian()
                                except Exception as e:
                                    st.error(
                                        f"Invalid date format in row {idx}: {j_date_str}"
                                    )
                                    continue
                                amount = float(row["amount"])
                                price = float(row["price"])
                                txn = {
                                    "jalali_date": j_date.strftime("%Y-%m-%d"),
                                    "gregorian_date": g_date.isoformat(),
                                    "amount": amount,
                                    "price": price,
                                    "total_cost": amount * price,
                                }
                                st.session_state["transactions"].append(txn)
                            st.success("CSV transactions imported successfully!")
                            st.rerun()
                    else:
                        st.error(
                            "CSV file is missing one or more required columns: 'jalali_date', 'amount', 'price'."
                        )
                except Exception as e:
                    st.error("Error reading CSV file. Please check its format.")

        st.markdown("### Add a New Transaction Manually")
        # -----------------------------------------------
        # Manual Transaction Entry Section
        # -----------------------------------------------
        # Jalali Date Picker using three selectboxes
        st.info("Select the purchase date using the Jalali (Persian) calendar.")
        today_jalali = jdatetime.date.today()
        years = list(range(1370, today_jalali.year + 1))
        default_year_index = (
            years.index(today_jalali.year)
            if today_jalali.year in years
            else len(years) - 1
        )
        selected_year = st.selectbox("Year (Jalali)", years, index=default_year_index)
        selected_month = st.selectbox(
            "Month (Jalali)", list(range(1, 13)), index=today_jalali.month - 1
        )
        # Determine maximum day for the selected month.
        if selected_month <= 6:
            max_day = 31
        elif selected_month <= 11:
            max_day = 30
        else:
            max_day = 30 if is_jalali_leap_year(selected_year) else 29
        days = list(range(1, max_day + 1))
        default_day_index = today_jalali.day - 1 if today_jalali.day <= max_day else 0
        selected_day = st.selectbox("Day (Jalali)", days, index=default_day_index)

        try:
            jalali_date = jdatetime.date(selected_year, selected_month, selected_day)
            gregorian_date = jalali_date.togregorian()
        except Exception as e:
            st.error("Invalid Jalali date selected.")
            return

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
                    "gregorian_date": gregorian_date.isoformat(),
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
            # List transactions with a delete button for each.
            for idx, txn in enumerate(st.session_state["transactions"]):
                cols = st.columns([1, 1, 1, 1, 1])
                cols[0].write(f"**Date:** {txn['jalali_date']}")
                cols[1].write(f"**BTC:** {txn['amount']:.8f}")
                cols[2].write(f"**Price:** ${txn['price']:,.2f}")
                cols[3].write(f"**Cost:** ${txn['total_cost']:,.2f}")
                if cols[4].button("Delete", key=f"delete_{idx}"):
                    st.session_state["transactions"].pop(idx)
                    st.rerun()

            # Also display a DataFrame of transactions.
            df = pd.DataFrame(st.session_state["transactions"])
            df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
            df = df.sort_values(by="greg_date")
            st.dataframe(df[["jalali_date", "amount", "price", "total_cost"]])

            # Basic Summary
            total_btc = df["amount"].sum()
            total_cost = df["total_cost"].sum()
            st.write(f"**Total BTC Purchased:** {total_btc:.8f}")
            st.write(f"**Total Invested:** ${total_cost:,.2f}")

            current_btc_price = display_price_section(st, "summary")
            if current_btc_price is not None:
                current_value = total_btc * current_btc_price
                profit = current_value - total_cost
                profit_percent = (profit / total_cost * 100) if total_cost > 0 else 0
                st.write(f"**Current Value of Holdings:** ${current_value:,.2f}")
                st.write(f"**Profit / Loss:** ${profit:,.2f} ({profit_percent:.2f}%)")

                # Bar chart: Invested vs. Current Value
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
            st.rerun()

    # ======================================================
    # TAB 2: ADVANCED ANALYTICS
    # ======================================================
    with tabs[1]:
        st.header("Advanced Analytics")

        if st.session_state["transactions"]:
            df = pd.DataFrame(st.session_state["transactions"])
            current_price = display_price_section(st, "analytics")

            if current_price:
                metrics = calculate_advanced_metrics(df, current_price)

                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Average Buy Price",
                        f"${metrics['average_buy_price']:,.2f}",
                        delta=f"${current_price - metrics['average_buy_price']:,.2f} vs Current",
                        help=f"Current BTC Price: ${current_price:,.2f}",
                    )
                    st.metric("Lowest Buy", f"${metrics['lowest_buy']:,.2f}")
                    st.metric(
                        "Total BTC",
                        f"{metrics['total_btc']:.8f}",
                        help="Total Bitcoin accumulated",
                    )

                with col2:
                    st.metric(
                        "Price Volatility",
                        f"${metrics['price_volatility']:,.2f}",
                        help="Standard deviation of your purchase prices",
                    )
                    st.metric("Highest Buy", f"${metrics['highest_buy']:,.2f}")
                    st.metric(
                        "Best Month",
                        metrics["best_month"],
                        help="Month with highest BTC accumulation",
                    )

                with col3:
                    st.metric(
                        "Total ROI",
                        f"{metrics['roi_percent']:.2f}%",
                        delta=f"${metrics['total_profit']:,.2f}",
                    )
                    st.metric(
                        "Monthly Avg Investment",
                        f"${metrics['highest_monthly_investment']:,.2f}",
                    )
                    st.metric(
                        "Current Value",
                        f"${metrics['current_value']:,.2f}",
                        help="Total holdings value at current price",
                    )

                # Add cost basis analysis
                st.subheader("Cost Basis Analysis")
                cost_basis_df = df.copy()
                cost_basis_df["greg_date"] = pd.to_datetime(
                    cost_basis_df["gregorian_date"]
                ).dt.date
                cost_basis_df = cost_basis_df.sort_values(
                    by="greg_date"
                )  # Sort by date
                cost_basis_df["cumulative_btc"] = cost_basis_df["amount"].cumsum()
                cost_basis_df["cumulative_cost"] = cost_basis_df["total_cost"].cumsum()
                cost_basis_df["average_cost"] = (
                    cost_basis_df["cumulative_cost"] / cost_basis_df["cumulative_btc"]
                )

                fig_cost = px.line(
                    cost_basis_df,
                    x="greg_date",
                    y="average_cost",
                    title="Average Cost Basis Over Time",
                    labels={"average_cost": "Cost Basis (USD)", "greg_date": "Date"},
                )
                st.plotly_chart(fig_cost, use_container_width=True)

                # Show monthly investment pattern
                st.subheader("Monthly Investment Pattern")
                monthly_data = df.copy()
                monthly_data["month"] = pd.to_datetime(
                    monthly_data["gregorian_date"]
                ).dt.strftime("%Y-%m")

                monthly_chart = (
                    monthly_data.groupby("month")
                    .agg({"total_cost": "sum", "amount": "sum", "price": "mean"})
                    .reset_index()
                )

                fig = px.bar(
                    monthly_chart,
                    x="month",
                    y="total_cost",
                    title="Monthly Investment Amount",
                    labels={
                        "month": "Month",
                        "total_cost": "Investment (USD)",
                    },
                    text=monthly_chart["total_cost"].apply(lambda x: f"${x:,.0f}"),
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_tickangle=-45, showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Add monthly statistics table
                st.subheader("Monthly Statistics")
                stats_df = monthly_chart.copy()
                stats_df["avg_price"] = stats_df["price"].apply(lambda x: f"${x:,.2f}")
                stats_df["total_invested"] = stats_df["total_cost"].apply(
                    lambda x: f"${x:,.2f}"
                )
                stats_df["btc_bought"] = stats_df["amount"].apply(lambda x: f"{x:.8f}")

                st.dataframe(
                    stats_df[
                        ["month", "total_invested", "btc_bought", "avg_price"]
                    ].rename(
                        columns={
                            "month": "Month",
                            "total_invested": "Total Invested",
                            "btc_bought": "BTC Bought",
                            "avg_price": "Average Price",
                        }
                    ),
                    hide_index=True,
                )

        else:
            st.info("Add transactions in the Summary tab to see advanced analytics.")

    # ======================================================
    # TAB 3: DCA PLANNING
    # ======================================================
    with tabs[2]:
        st.header("Dollar-Cost Averaging (DCA) Calculator")

        col1, col2 = st.columns(2)

        with col1:
            investment_amount = st.number_input(
                "Regular Investment Amount (USD)", min_value=1.0, value=100.0, step=10.0
            )

            frequency = st.selectbox(
                "Investment Frequency", ["Daily", "Weekly", "Monthly"]
            )

        with col2:
            duration = st.slider(
                "Planning Duration (months)", min_value=1, max_value=60, value=12
            )

            # Updated growth rate input with helpful context
            st.markdown(
                """
                <style>
                .tooltip {
                    color: #666;
                    font-size: 0.85em;
                    font-style: italic;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                ##### Expected Annual BTC Growth (%)
                <div class="tooltip">
                Default: 20% - Based on conservative long-term BTC projections.
                Historical average: ~100% per year (highly volatile).
                Adjust based on your market outlook.
                </div>
            """,
                unsafe_allow_html=True,
            )

            expected_growth = st.slider(
                "Select growth rate",
                min_value=-50,
                max_value=200,
                value=20,  # Changed default to 20%
                help="Conservative estimate based on long-term BTC projections. Historical performance doesn't guarantee future returns.",
            )

        # Calculate DCA projections
        current_price = display_price_section(st, "dca")
        if current_price:
            # Calculate number of periods and adjust growth rate based on frequency
            if frequency == "Daily":
                periods = duration * 30
                periodic_growth = (1 + expected_growth / 100) ** (
                    1 / (12 * 30)
                ) - 1  # Daily growth rate
            elif frequency == "Weekly":
                periods = duration * 4
                periodic_growth = (1 + expected_growth / 100) ** (
                    1 / (12 * 4)
                ) - 1  # Weekly growth rate
            else:  # Monthly
                periods = duration
                periodic_growth = (1 + expected_growth / 100) ** (
                    1 / 12
                ) - 1  # Monthly growth rate

            projection_data = []
            total_invested = 0
            total_btc = 0
            current_btc_price = current_price

            for period in range(periods):
                # Update BTC price for this period
                current_btc_price = current_btc_price * (1 + periodic_growth)

                # Calculate investment for this period
                btc_bought = investment_amount / current_btc_price
                total_invested += investment_amount
                total_btc += btc_bought

                projection_data.append(
                    {
                        "period": period + 1,
                        "total_invested": total_invested,
                        "total_btc": total_btc,
                        "estimated_value": total_btc * current_btc_price,
                        "btc_price": current_btc_price,
                    }
                )

            projection_df = pd.DataFrame(projection_data)

            st.subheader("DCA Projection")
            fig = px.line(
                projection_df,
                x="period",
                y=["total_invested", "estimated_value"],
                title="Investment Growth Projection",
                labels={
                    "value": "USD",
                    "variable": "Metric",
                    "period": f"Number of {frequency.lower()} periods",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Final BTC price is the last calculated price
            final_btc_price = projection_df["btc_price"].iloc[-1]

            # Update metrics display with estimated final BTC price
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Projected Final BTC Holdings",
                    f"{total_btc:.8f} BTC",
                    help=f"Estimated BTC price: ${final_btc_price:,.2f}",
                )
            with col2:
                st.metric(
                    "Projected Final Value",
                    f"${projection_df['estimated_value'].iloc[-1]:,.2f}",
                )

            st.metric(
                "Projected Regular Investment",
                f"${projection_df['total_invested'].iloc[-1]:,.2f}",
                delta=f"${projection_df['estimated_value'].iloc[-1] - projection_df['total_invested'].iloc[-1]:,.2f}",
                delta_color="normal",
            )
        else:
            st.error("Unable to fetch current BTC price for calculations.")

    # ======================================================
    # TAB 4: TIMELINE ANALYSIS (Advanced Charts)
    # ======================================================
    with tabs[3]:
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

            # Build a daily timeline.
            timeline = []
            cumulative_btc = 0.0
            cumulative_cost = 0.0
            txn_idx = 0
            txns = df.to_dict("records")
            num_txns = len(txns)
            current_day = earliest_date

            while current_day <= today:
                # Add any transactions for the current day.
                while txn_idx < num_txns and txns[txn_idx]["greg_date"] == current_day:
                    cumulative_btc += txns[txn_idx]["amount"]
                    cumulative_cost += txns[txn_idx]["total_cost"]
                    txn_idx += 1

                # Fetch the BTC price for the current day
                btc_price = price_by_date.get(current_day)

                # If the price is not available, use the last known price or fetch the current price
                if btc_price is None:
                    if timeline:
                        btc_price = timeline[-1]["btc_price"]
                    else:
                        btc_price = get_current_btc_price()

                # Ensure btc_price is not None
                if btc_price is None:
                    st.error(
                        "Unable to fetch BTC price for some dates. Please check your internet connection or try again later."
                    )
                    return

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

    # ======================================================
    # TAB 5: EXPORT DATA
    # ======================================================
    with tabs[4]:
        st.header("Export Your Data")

        if st.session_state["transactions"]:
            st.info("Download your transaction history in different formats.")

            col1, col2 = st.columns(2)

            with col1:
                # Excel Export
                excel_data = export_to_excel(st.session_state["transactions"])
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name="btc_transactions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col2:
                # CSV Export
                df = pd.DataFrame(st.session_state["transactions"])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="btc_transactions.csv",
                    mime="text/csv",
                )

            # Preview of export data
            st.subheader("Data Preview")
            st.dataframe(
                pd.DataFrame(st.session_state["transactions"])[
                    ["jalali_date", "amount", "price", "total_cost"]
                ]
            )
        else:
            st.info("Add transactions in the Summary tab to enable data export.")


if __name__ == "__main__":
    main()
