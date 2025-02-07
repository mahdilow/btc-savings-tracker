import streamlit as st
import pandas as pd
import requests
import datetime


def get_current_btc_price():
    """
    Fetch the current BTC price in USD using CoinGecko's API.
    Returns the price as a float or None if there is an error.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            price = data["bitcoin"]["usd"]
            return price
        else:
            return None
    except Exception as e:
        st.error("Error fetching current BTC price.")
        return None


def main():
    st.title("Bitcoin Savings & Profit Tracker")
    st.write("Enter your Bitcoin purchase details to see your investment analysis.")

    # Initialize a session state variable to store transactions.
    if "transactions" not in st.session_state:
        st.session_state["transactions"] = []

    st.markdown("## Add a New Transaction")
    # Use a form so the user can input a transaction and submit it.
    with st.form(key="transaction_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        # Date input defaults to today's date.
        date_input = col1.date_input("Purchase Date", value=datetime.date.today())
        # The BTC amount (you can enter very small fractions).
        amount = col2.number_input(
            "BTC Amount", min_value=0.0, format="%.8f", step=0.00000001
        )
        # Price per BTC in USD.
        price = col3.number_input(
            "Price per BTC (USD)", min_value=0.0, format="%.2f", step=0.01
        )

        # Button to submit the form.
        submit_button = st.form_submit_button(label="Add Transaction")

        if submit_button:
            # Save the transaction in the session state.
            st.session_state["transactions"].append(
                {
                    "date": date_input.strftime("%Y-%m-%d"),
                    "amount": amount,
                    "price": price,
                    "total_cost": amount * price,
                }
            )
            st.success("Transaction added!")

    st.markdown("## Your Transactions")
    # If there are any transactions added, show them and compute analytics.
    if st.session_state["transactions"]:
        # Create a DataFrame from the stored transactions.
        df = pd.DataFrame(st.session_state["transactions"])
        # Sort transactions by date for clarity.
        df = df.sort_values(by="date")
        st.dataframe(df)

        # Compute overall stats.
        total_btc = df["amount"].sum()
        total_cost = df["total_cost"].sum()

        st.write(f"**Total BTC Purchased:** {total_btc:.8f}")
        st.write(f"**Total Invested:** ${total_cost:,.2f}")

        # Get the current BTC price from the API.
        current_btc_price = get_current_btc_price()
        if current_btc_price is not None:
            st.write(f"**Current BTC Price:** ${current_btc_price:,.2f}")
            current_value = total_btc * current_btc_price
            profit = current_value - total_cost
            profit_percent = (profit / total_cost * 100) if total_cost > 0 else 0
            st.write(f"**Current Value of Holdings:** ${current_value:,.2f}")
            st.write(f"**Profit / Loss:** ${profit:,.2f} ({profit_percent:.2f}%)")
        else:
            st.error("Could not fetch current BTC price. Please try again later.")
            # Fallback: set current_value equal to total_cost so the chart still shows something.
            current_value = total_cost

        # Plot a simple bar chart comparing the invested amount and current value.
        st.markdown("### Investment Comparison")
        chart_data = pd.DataFrame(
            {
                "Metric": ["Invested", "Current Value"],
                "USD": [total_cost, current_value],
            }
        )
        st.bar_chart(chart_data.set_index("Metric"))

        # Option to clear all transactions.
        if st.button("Clear All Transactions"):
            st.session_state["transactions"] = []
            st.experimental_rerun()
    else:
        st.info("No transactions added yet. Please add a transaction above.")


if __name__ == "__main__":
    main()
