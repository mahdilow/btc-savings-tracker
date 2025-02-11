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
def get_historical_btc_prices(
    from_date: datetime.date,
    to_date: datetime.date,
    max_retries: int = 3,
    retry_delay: float = 60.0,  # Increased to 60 seconds
) -> dict:
    """
    Fetch historical BTC prices with improved rate limit handling
    """
    # Split date range into 90-day chunks to avoid rate limits
    price_by_date = {}
    chunk_size = datetime.timedelta(days=90)
    current_from = from_date

    while current_from <= to_date:
        current_to = min(current_from + chunk_size, to_date)

        from_ts = int(
            datetime.datetime.combine(current_from, datetime.time()).timestamp()
        )
        to_ts = int(datetime.datetime.combine(current_to, datetime.time()).timestamp())

        for attempt in range(max_retries):
            try:
                url = (
                    f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
                    f"?vs_currency=usd&from={from_ts}&to={to_ts}"
                )
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    prices = data.get("prices", [])
                    for ts, price in prices:
                        dt = datetime.datetime.fromtimestamp(ts / 1000).date()
                        price_by_date[dt] = price
                    break  # Success, move to next chunk
                elif response.status_code == 429:
                    print(f"Rate limit hit, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(
                        f"Attempt {attempt + 1} failed with status code: {response.status_code}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(
                            retry_delay / 3
                        )  # Shorter delay for non-rate-limit errors
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay / 3)

        # Move to next chunk
        current_from = current_to + datetime.timedelta(days=1)
        time.sleep(1)  # Small delay between chunks

    return price_by_date


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
    workbook = xlsxwriter.Workbook(output, {"nan_inf_to_errors": True})
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
        worksheet.write(row, 2, float(txn["amount"]) if pd.notna(txn["amount"]) else 0)
        worksheet.write(row, 3, float(txn["price"]) if pd.notna(txn["price"]) else 0)
        worksheet.write(
            row, 4, float(txn["total_cost"]) if pd.notna(txn["total_cost"]) else 0
        )

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


def initialize_price_data():
    """
    Initialize both current and historical price data
    Returns True if initialization was successful
    """
    success = True

    # Initialize current price if not already done
    if "current_btc_price" not in st.session_state:
        with st.spinner("Fetching current BTC price..."):
            price = get_current_btc_price()
            if price:
                st.session_state["current_btc_price"] = price
                st.session_state["price_fetch_time"] = datetime.datetime.now().strftime(
                    "%H:%M:%S"
                )
            else:
                success = False

    # Initialize historical prices if we have transactions
    if (
        "transactions" in st.session_state
        and st.session_state["transactions"]
        and (
            "historical_prices" not in st.session_state
            or not st.session_state["historical_prices"]
        )
    ):
        try:
            with st.spinner(
                "Fetching historical BTC prices (this may take a few minutes)..."
            ):
                df = pd.DataFrame(st.session_state["transactions"])
                df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
                earliest_date = df["greg_date"].min()
                today = datetime.date.today()

                prices = get_historical_btc_prices(earliest_date, today)
                if prices and len(prices) > 0:
                    st.session_state["historical_prices"] = prices
                    st.session_state["historical_price_fetch_time"] = (
                        datetime.datetime.now().strftime("%H:%M:%S")
                    )
                else:
                    success = False

        except Exception as e:
            success = False
            st.error(f"Error initializing historical prices: {str(e)}")

    return success


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


def display_historical_price_section(location, df: pd.DataFrame, section_name: str):
    """
    Display historical price section with retry button
    Returns the historical prices dictionary or empty dict if unavailable
    """
    # First check if we have valid historical prices in session state
    if (
        "historical_prices" in st.session_state
        and st.session_state["historical_prices"]
        and isinstance(st.session_state["historical_prices"], dict)
        and len(st.session_state["historical_prices"]) > 0
    ):
        return st.session_state["historical_prices"]

    # If we don't have valid prices, show error and retry button
    col1, col2 = location.columns([3, 1])
    col1.error(
        "Unable to fetch historical BTC prices. Please check your internet connection."
    )
    if col2.button("🔄 Retry History", key=f"retry_history_{section_name}"):
        earliest_date = df["greg_date"].min()
        today = datetime.date.today()
        prices = get_historical_btc_prices(earliest_date, today)
        if prices and len(prices) > 0:  # Verify we got valid data
            st.session_state["historical_prices"] = prices
            st.session_state["historical_price_fetch_time"] = (
                datetime.datetime.now().strftime("%H:%M:%S")
            )
            st.rerun()
    return {}


# -------------------------------------------------------
# Main Application
# -------------------------------------------------------


def main():
    st.title("Bitcoin Savings & Profit Tracker")

    # Add calendar preference in sidebar
    with st.sidebar:
        st.header("Settings")
        if "calendar_type" not in st.session_state:
            st.session_state["calendar_type"] = "Gregorian"

        calendar_type = st.selectbox(
            "Select Calendar Type",
            ["Gregorian", "Jalali (Persian)"],
            key="calendar_type",
        )
        st.caption("You can change the calendar type at any time.")

    st.write(
        "Track your BTC purchases, analyze your overall investment, and view timeline analytics."
    )

    # Initialize prices at startup
    initialize_price_data()

    # Show current price fetch time
    if "price_fetch_time" in st.session_state:
        st.caption(
            f"Current price last fetched at: {st.session_state['price_fetch_time']}"
        )

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
            if st.session_state["calendar_type"] == "Jalali (Persian)":
                date_column = "jalali_date"
                date_format = "YYYY-MM-DD"
                example_date = "1402-01-15"
            else:
                date_column = "date"
                date_format = "YYYY-MM-DD"
                example_date = "2024-01-15"

            st.write(
                f"Upload a CSV file containing your BTC purchase transactions. Follow these guidelines to ensure successful import:"
            )

            st.markdown(
                f"""
                ### Required CSV Format:
                
                1. **Column Names (Header Row):**
                   - `{date_column}`: Purchase date
                   - `amount`: BTC amount purchased
                   - `price`: Price per BTC in USD

                2. **Data Format Requirements:**
                   - **{date_column}:** {date_format} (example: {example_date})
                   - **amount:** Decimal number (example: 0.00123456)
                   - **price:** USD amount without $ symbol (example: 42000)
                
                ### Example CSV Content:
                ```
                {date_column},amount,price
                {example_date},0.00123456,42000
                {example_date},0.00050000,43500
                ```

                ### Tips:
                - Use comma (,) as the separator
                - Don't include $ symbols in prices
                - Don't include commas in numbers
                - Dates must be in exact format: {date_format}
                """
            )

            # Add a downloadable example CSV
            example_data = f"""{date_column},amount,price
{example_date},0.00123456,42000
{example_date},0.00050000,43500"""

            st.download_button(
                "📥 Download Example CSV",
                example_data,
                "example_transactions.csv",
                "text/csv",
                help="Download a sample CSV file with the correct format",
            )

            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)
                    required_cols = [date_column, "amount", "price"]
                    if all(col in csv_df.columns for col in required_cols):
                        if st.button("Import CSV Transactions"):
                            for idx, row in csv_df.iterrows():
                                date_str = str(row[date_column])
                                try:
                                    # Parse date based on calendar type
                                    if (
                                        st.session_state["calendar_type"]
                                        == "Jalali (Persian)"
                                    ):
                                        # Handle Jalali date
                                        year, month, day = map(int, date_str.split("-"))
                                        j_date = jdatetime.date(year, month, day)
                                        g_date = j_date.togregorian()
                                        jalali_str = j_date.strftime("%Y-%m-%d")
                                    else:
                                        # Handle Gregorian date
                                        g_date = datetime.datetime.strptime(
                                            date_str, "%Y-%m-%d"
                                        ).date()
                                        j_date = jdatetime.date.fromgregorian(
                                            date=g_date
                                        )
                                        jalali_str = j_date.strftime("%Y-%m-%d")

                                    amount = float(row["amount"])
                                    price = float(row["price"])

                                    # Validate date range
                                    if g_date > datetime.date.today():
                                        st.warning(
                                            f"Skipping future date in row {idx + 1}: {date_str}"
                                        )
                                        continue
                                    if g_date.year < 1990:
                                        st.warning(
                                            f"Skipping date before 1990 in row {idx + 1}: {date_str}"
                                        )
                                        continue

                                    txn = {
                                        "jalali_date": jalali_str,
                                        "gregorian_date": g_date.isoformat(),
                                        "amount": amount,
                                        "price": price,
                                        "total_cost": amount * price,
                                    }
                                    st.session_state["transactions"].append(txn)
                                except ValueError as e:
                                    st.error(
                                        f"Invalid date format in row {idx + 1}: {date_str}"
                                    )
                                    continue
                                except Exception as e:
                                    st.error(
                                        f"Error processing row {idx + 1}: {str(e)}"
                                    )
                                    continue

                            st.success("CSV transactions imported successfully!")
                            st.rerun()
                    else:
                        st.error(
                            f"CSV file is missing one or more required columns: '{date_column}', 'amount', 'price'."
                        )
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")

        st.markdown("### Add a New Transaction Manually")
        # -----------------------------------------------
        # Manual Transaction Entry Section
        # -----------------------------------------------
        if st.session_state["calendar_type"] == "Jalali (Persian)":
            st.info("Select the purchase date using the Jalali (Persian) calendar.")
            today_jalali = jdatetime.date.today()
            years = list(range(1370, today_jalali.year + 1))
            default_year_index = (
                years.index(today_jalali.year)
                if today_jalali.year in years
                else len(years) - 1
            )
            selected_year = st.selectbox(
                "Year (Jalali)", years, index=default_year_index
            )
            selected_month = st.selectbox(
                "Month (Jalali)", list(range(1, 13)), index=today_jalali.month - 1
            )

            # Determine maximum day for Jalali calendar
            if selected_month <= 6:
                max_day = 31
            elif selected_month <= 11:
                max_day = 30
            else:
                max_day = 30 if is_jalali_leap_year(selected_year) else 29

            days = list(range(1, max_day + 1))
            default_day_index = (
                today_jalali.day - 1 if today_jalali.day <= max_day else 0
            )
            selected_day = st.selectbox("Day (Jalali)", days, index=default_day_index)

            try:
                selected_date = jdatetime.date(
                    selected_year, selected_month, selected_day
                )
                gregorian_date = selected_date.togregorian()
                jalali_date = selected_date
                display_date = selected_date.strftime("%Y-%m-%d")
            except Exception as e:
                st.error("Invalid Jalali date selected.")
                return
        else:
            st.info("Select the purchase date using the Gregorian calendar.")
            today = datetime.date.today()
            selected_date = st.date_input(
                "Purchase Date",
                value=today,
                min_value=datetime.date(1990, 1, 1),
                max_value=today,
            )
            gregorian_date = selected_date
            try:
                jalali_date = jdatetime.date.fromgregorian(date=selected_date)
                display_date = selected_date.strftime("%Y-%m-%d")
            except Exception as e:
                st.error("Error converting date.")
                return

        # Rest of the transaction entry code
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

        st.markdown("### Transaction History")
        if st.session_state["transactions"]:
            try:
                df = pd.DataFrame(st.session_state["transactions"])

                # Determine which date column to show based on calendar preference
                display_columns = ["amount", "price", "total_cost"]
                if st.session_state["calendar_type"] == "Jalali (Persian)":
                    date_col = "jalali_date"
                    date_label = "Jalali Date"
                else:
                    date_col = "gregorian_date"
                    date_label = "Date"

                display_columns.insert(0, date_col)

                # Sort by gregorian date internally
                df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
                df = df.sort_values(by="greg_date")

                # Format the display data
                display_df = df[display_columns].copy()

                # Rename columns for display
                column_labels = {
                    date_col: date_label,
                    "amount": "BTC Amount",
                    "price": "Price (USD)",
                    "total_cost": "Total Cost (USD)",
                }

                formatted_df = display_df.rename(columns=column_labels)

                # Format numeric columns
                formatted_df["BTC Amount"] = formatted_df["BTC Amount"].apply(
                    lambda x: f"{x:.8f}"
                )
                formatted_df["Price (USD)"] = formatted_df["Price (USD)"].apply(
                    lambda x: f"${x:,.2f}"
                )
                formatted_df["Total Cost (USD)"] = formatted_df[
                    "Total Cost (USD)"
                ].apply(lambda x: f"${x:,.2f}")

                st.dataframe(
                    formatted_df,
                    hide_index=True,
                    use_container_width=True,
                )

                # Add helpful tips for data interpretation
                with st.expander("📊 Data Grid Tips"):
                    st.markdown(
                        """
                    ### Understanding Your Transaction Data
                    
                    - **Date Format:** Dates are shown in {cal_type} calendar format (YYYY-MM-DD)
                    - **BTC Amount:** Shows exact amount to 8 decimal places
                    - **Price:** The price per BTC in USD at time of purchase
                    - **Total Cost:** Amount × Price
                    
                    **Tips:**
                    - Click column headers to sort
                    - Use the search box to filter data
                    - Right-click for additional options
                    
                    **Note:** All monetary values are in USD
                    """.format(
                            cal_type=(
                                "Jalali"
                                if st.session_state["calendar_type"]
                                == "Jalali (Persian)"
                                else "Gregorian"
                            )
                        )
                    )

            except Exception as e:
                st.error(f"Error displaying transaction data: {str(e)}")
                st.info(
                    "Try refreshing the page. If the problem persists, check your transaction data for any inconsistencies."
                )

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
    # TAB 4: TIMELINE ANALYSIS
    # ======================================================
    with tabs[3]:
        st.header("Timeline Analysis")
        if st.session_state["transactions"]:
            # Initialize historical prices if needed
            if initialize_price_data():
                # Get historical prices from session state
                price_by_date = st.session_state.get("historical_prices", {})

                if price_by_date and len(price_by_date) > 0:
                    # Show fetch time if available
                    if "historical_price_fetch_time" in st.session_state:
                        st.caption(
                            f"Historical prices last fetched at: {st.session_state['historical_price_fetch_time']}"
                        )

                    # Create and prepare DataFrame
                    df = pd.DataFrame(st.session_state["transactions"])
                    df["greg_date"] = pd.to_datetime(df["gregorian_date"]).dt.date
                    df = df.sort_values(by="greg_date")

                    # Build a daily timeline.
                    timeline = []
                    cumulative_btc = 0.0
                    cumulative_cost = 0.0
                    txn_idx = 0
                    txns = df.to_dict("records")
                    num_txns = len(txns)
                    current_day = df["greg_date"].min()

                    while current_day <= datetime.date.today():
                        # Add any transactions for the current day.
                        while (
                            txn_idx < num_txns
                            and txns[txn_idx]["greg_date"] == current_day
                        ):
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
                    st.error("Unable to fetch historical BTC prices. Please try again.")
                    if st.button("🔄 Retry"):
                        st.session_state.pop("historical_prices", None)
                        st.rerun()
            else:
                st.error(
                    "Unable to fetch price data. Please check your internet connection."
                )
                if st.button("🔄 Retry"):
                    st.session_state.pop("historical_prices", None)
                    st.session_state.pop("current_btc_price", None)
                    st.rerun()
        else:
            st.info("Add transactions in the Summary tab to see timeline analysis.")
            st.info("Add transactions in the Summary tab to enable data export.")

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

            # Determine which date column to show based on calendar preference
            if st.session_state["calendar_type"] == "Jalali (Persian)":
                date_col = "jalali_date"
            else:
                date_col = "gregorian_date"

            preview_df = pd.DataFrame(st.session_state["transactions"])
            st.dataframe(
                preview_df[[date_col, "amount", "price", "total_cost"]].rename(
                    columns={
                        date_col: "Date",
                        "amount": "BTC Amount",
                        "price": "Price (USD)",
                        "total_cost": "Total Cost (USD)",
                    }
                ),
                hide_index=True,
            )
        else:
            st.info("Add transactions in the Summary tab to enable data export.")


if __name__ == "__main__":
    main()
