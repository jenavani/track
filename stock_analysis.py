import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
from tabulate import tabulate
import sys
import re
import warnings
warnings.filterwarnings('ignore')

class TradingAnalysis:
    def __init__(self):
        self.df = None
        self.stock_transactions = None
        self.option_transactions = None
        self.other_transactions = None
        self.results = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(file_path)
            # Convert date columns to datetime
            for date_col in ['Activity Date', 'Process Date', 'Settle Date']:
                if date_col in self.df.columns:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
            
            # Process amounts - remove $ and convert to float
            if 'Amount' in self.df.columns:
                self.df['Amount'] = self.df['Amount'].replace(r'[\$,)]', '', regex=True)
                self.df['Amount'] = self.df['Amount'].replace(r'[(]', '-', regex=True)
                self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
            
            print(f"Successfully loaded {len(self.df)} transactions from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def categorize_transactions(self):
        """Categorize transactions into Stock, Options, and Other"""
        # Create new column to categorize transactions
        self.df['Category'] = 'Other'
        
        # Identify option transactions
        option_pattern = r'(Call|Put) \$\d+\.\d+'
        self.df.loc[self.df['Description'].str.contains(option_pattern, na=False), 'Category'] = 'Options'
        
        # Mark all transactions with STO, BTO, STC, BTC trans codes as Options
        option_codes = ['STO', 'BTO', 'STC', 'BTC', 'OEXP', 'OASGN']
        self.df.loc[self.df['Trans Code'].isin(option_codes), 'Category'] = 'Options'
        
        # Identify stock transactions
        stock_codes = ['Buy', 'Sell']
        # Exclude option assignments from stock category
        self.df.loc[(self.df['Trans Code'].isin(stock_codes)) & 
                    (~self.df['Description'].str.contains('Options Assigned', na=False, case=False)), 'Category'] = 'Stock'
        
        # Split into separate dataframes
        self.stock_transactions = self.df[self.df['Category'] == 'Stock'].copy()
        self.option_transactions = self.df[self.df['Category'] == 'Options'].copy()
        self.other_transactions = self.df[self.df['Category'] == 'Other'].copy()
        
        print(f"Categorized transactions: {len(self.stock_transactions)} Stock, "
              f"{len(self.option_transactions)} Options, {len(self.other_transactions)} Other")
        
    def extract_option_details(self):
        """Extract option details from the description"""
        # Add columns for option details
        self.option_transactions['Option Type'] = None
        self.option_transactions['Strike Price'] = None
        self.option_transactions['Expiration Date'] = None
        self.option_transactions['Option ID'] = None
        
        # Extract option details from Description
        for idx, row in self.option_transactions.iterrows():
            desc = row['Description'] if isinstance(row['Description'], str) else ''
            
            # Extract expiration date
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', desc)
            if date_match:
                self.option_transactions.at[idx, 'Expiration Date'] = date_match.group(1)
            
            # Extract option type (Call/Put)
            type_match = re.search(r'(Call|Put)', desc)
            if type_match:
                self.option_transactions.at[idx, 'Option Type'] = type_match.group(1)
            
            # Extract strike price
            price_match = re.search(r'\$(\d+\.\d+)', desc)
            if price_match:
                self.option_transactions.at[idx, 'Strike Price'] = price_match.group(1)
        
        # Create a unique ID for each option
        self.option_transactions['Option ID'] = (
            self.option_transactions['Instrument'] + '_' + 
            self.option_transactions['Expiration Date'].astype(str) + '_' + 
            self.option_transactions['Option Type'].astype(str) + '_' + 
            self.option_transactions['Strike Price'].astype(str)
        )
    
    def calculate_stock_pnl(self):
        """Calculate profit/loss for stock transactions"""
        # Initialize columns
        self.stock_transactions['Realized_Profit_Loss'] = 0.0
        self.stock_transactions['Unrealized_Profit_Loss'] = 0.0
        
        # Group by stock symbol
        stock_groups = self.stock_transactions.groupby('Instrument')
        
        results = []
        
        for symbol, group in stock_groups:
            if symbol is None or pd.isna(symbol):
                continue
                
            # Sort by date
            sorted_group = group.sort_values('Activity Date')
            
            # Calculate cumulative shares
            sorted_group['Running_Shares'] = sorted_group['Quantity'].cumsum()
            
            # Calculate cost basis
            sorted_group['Cost_Basis'] = 0.0
            cost_basis = 0.0
            shares_owned = 0
            realized_pnl = 0.0
            
            for idx, row in sorted_group.iterrows():
                trans_code = row['Trans Code']
                quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                price = row['Price'] if not pd.isna(row['Price']) else 0
                amount = row['Amount'] if not pd.isna(row['Amount']) else 0
                
                if trans_code == 'Buy':
                    cost_basis += abs(amount)
                    shares_owned += quantity
                    if idx in self.df.index:
                        self.df.at[idx, 'Cost_Basis'] = cost_basis
                elif trans_code == 'Sell':
                    if shares_owned > 0:
                        # Calculate realized profit/loss
                        sell_value = abs(amount)
                        avg_cost = cost_basis / shares_owned if shares_owned > 0 else 0
                        realized_pnl_for_sale = sell_value - (quantity * avg_cost)
                        realized_pnl += realized_pnl_for_sale
                        if idx in self.df.index:
                            self.df.at[idx, 'Realized_Profit_Loss'] = realized_pnl_for_sale
                        
                        # Update cost basis
                        cost_basis -= (quantity * avg_cost)
                        shares_owned -= quantity
            
            # Calculate unrealized P&L if there are shares left
            unrealized_pnl = 0.0
            if shares_owned > 0 and cost_basis > 0:
                try:
                    # Get current stock price
                    stock_data = yf.Ticker(symbol)
                    current_price = stock_data.history(period="1d")['Close'].iloc[-1]
                    unrealized_pnl = (current_price * shares_owned) - cost_basis
                except Exception as e:
                    # Use the last known price as fallback
                    last_price = sorted_group['Price'].iloc[-1] if len(sorted_group['Price']) > 0 and not sorted_group['Price'].empty else 0
                    unrealized_pnl = (last_price * shares_owned) - cost_basis
            
            results.append({
                'Symbol': symbol,
                'Transaction_Type': 'Stock',
                'Realized_Profit_Loss': realized_pnl,
                'Unrealized_Profit_Loss': unrealized_pnl,
                'Shares_Owned': shares_owned,
                'Cost_Basis': cost_basis
            })
        
        self.stock_results = pd.DataFrame(results)
        return self.stock_results
    
    def calculate_option_pnl(self):
        """Calculate profit/loss for option transactions"""
        # Extract option details first
        self.extract_option_details()
        
        # Initialize columns if they don't exist
        if 'Realized_Profit_Loss' not in self.option_transactions.columns:
            self.option_transactions['Realized_Profit_Loss'] = 0.0
        if 'Unrealized_Profit_Loss' not in self.option_transactions.columns:
            self.option_transactions['Unrealized_Profit_Loss'] = 0.0
        
        # Group by option ID
        option_groups = self.option_transactions.groupby(['Instrument', 'Option ID'])
        
        results = []
        
        for (symbol, option_id), group in option_groups:
            if symbol is None or pd.isna(symbol) or option_id is None or pd.isna(option_id):
                continue
                
            # Sort by date
            sorted_group = group.sort_values('Activity Date')
            
            # Get option details from the first row
            first_row = sorted_group.iloc[0]
            option_type = first_row['Option Type']
            strike_price = first_row['Strike Price']
            expiration_date = first_row['Expiration Date']
            
            # Track contracts and amounts
            contracts_open = 0
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            
            for idx, row in sorted_group.iterrows():
                trans_code = row['Trans Code']
                quantity = row['Quantity'] if not pd.isna(row['Quantity']) else 0
                amount = row['Amount'] if not pd.isna(row['Amount']) else 0
                
                # Option transactions
                if trans_code == 'STO':  # Sell to Open
                    contracts_open += quantity
                    realized_pnl += amount  # Credit received
                elif trans_code == 'BTO':  # Buy to Open
                    contracts_open += quantity
                    realized_pnl -= abs(amount)  # Debit paid
                elif trans_code == 'STC':  # Sell to Close
                    contracts_open -= quantity
                    realized_pnl += amount  # Credit received
                elif trans_code == 'BTC':  # Buy to Close
                    contracts_open -= quantity
                    realized_pnl -= abs(amount)  # Debit paid
                elif trans_code == 'OEXP':  # Option Expiration
                    # For expired options, the value is 0
                    contracts_open = 0
                elif trans_code == 'OASGN':  # Option Assignment
                    # For assigned options, handle separately with the stock transaction
                    contracts_open = 0
            
            # Calculate unrealized P&L if there are contracts left
            if contracts_open != 0:
                try:
                    # Try to get current option price (this is simplified and may not work for all options)
                    try:
                        # Format the expiration date string
                        exp_date_obj = datetime.strptime(expiration_date, "%m/%d/%Y")
                        exp_date_formatted = exp_date_obj.strftime("%y%m%d")
                        
                        # Format the strike price
                        strike_formatted = str(float(strike_price) * 1000).split('.')[0].zfill(8)
                        
                        # Construct the OCC option symbol
                        option_symbol = f"{symbol}{exp_date_formatted}{'C' if option_type == 'Call' else 'P'}{strike_formatted}"
                        option_data = yf.Ticker(option_symbol)
                        current_price = option_data.history(period="1d")['Close'].iloc[-1]
                        unrealized_pnl = current_price * contracts_open * 100  # Options are for 100 shares
                    except Exception:
                        # Fallback to another format
                        print(f"Could not get current price for {option_id}, using last known value.")
                        last_price = sorted_group['Price'].iloc[-1] if not sorted_group['Price'].empty else 0
                        unrealized_pnl = last_price * contracts_open * 100
                except Exception as e:
                    # Fallback - just use the last known price
                    last_price = sorted_group['Price'].iloc[-1] if not sorted_group['Price'].empty else 0
                    unrealized_pnl = last_price * contracts_open * 100
            
            results.append({
                'Symbol': symbol,
                'Transaction_Type': 'Options',
                'Option_ID': option_id,
                'Realized_Profit_Loss': realized_pnl,
                'Unrealized_Profit_Loss': unrealized_pnl,
                'Contracts_Open': contracts_open
            })
        
        self.option_results = pd.DataFrame(results)
        return self.option_results
    
    def combine_results(self):
        """Combine stock and option results into a single DataFrame"""
        # Calculate P&L for stocks and options
        stock_results = self.calculate_stock_pnl()
        option_results = self.calculate_option_pnl()
        
        # Combine results
        combined_results = []
        
        if not stock_results.empty:
            for _, row in stock_results.iterrows():
                combined_results.append({
                    'Symbol': row['Symbol'],
                    'Transaction_Type': 'Stock',
                    'Realized_Profit_Loss': row['Realized_Profit_Loss'],
                    'Unrealized_Profit_Loss': row['Unrealized_Profit_Loss']
                })
        
        if not option_results.empty:
            for _, row in option_results.iterrows():
                combined_results.append({
                    'Symbol': row['Symbol'],
                    'Transaction_Type': 'Options',
                    'Realized_Profit_Loss': row['Realized_Profit_Loss'],
                    'Unrealized_Profit_Loss': row['Unrealized_Profit_Loss']
                })
        
        self.results = pd.DataFrame(combined_results)
        return self.results
    
    def filter_by_period(self, period):
        """Filter results by the specified period"""
        today = datetime.now()
        
        if period == 'Current Month':
            start_date = datetime(today.year, today.month, 1)
            end_date = today
        elif period == 'Last Month':
            if today.month == 1:
                start_date = datetime(today.year - 1, 12, 1)
                end_date = datetime(today.year, today.month, 1) - timedelta(days=1)
            else:
                start_date = datetime(today.year, today.month - 1, 1)
                end_date = datetime(today.year, today.month, 1) - timedelta(days=1)
        elif period == 'Year-to-date':
            start_date = datetime(today.year, 1, 1)
            end_date = today
        else:  # 'All'
            return self.results
        
        # Filter transactions by date range
        filtered_df = self.df[
            (self.df['Activity Date'] >= pd.Timestamp(start_date)) & 
            (self.df['Activity Date'] <= pd.Timestamp(end_date))
        ].copy()
        
        # Create temporary analysis object with filtered data
        temp_analysis = TradingAnalysis()
        temp_analysis.df = filtered_df
        temp_analysis.categorize_transactions()
        
        # Get filtered results
        filtered_results = temp_analysis.combine_results()
        return filtered_results
    
    def display_results(self, period='All'):
        """Display results in a table format for the specified period"""
        if self.results is None:
            self.combine_results()
        
        filtered_results = self.filter_by_period(period)
        
        if filtered_results.empty:
            print(f"No transactions found for the period: {period}")
            return
        
        # Group by Symbol and Transaction_Type
        grouped = filtered_results.groupby(['Symbol', 'Transaction_Type']).agg({
            'Realized_Profit_Loss': 'sum',
            'Unrealized_Profit_Loss': 'sum'
        }).reset_index()
        
        # Format the results
        formatted_results = grouped.copy()
        formatted_results['Realized_Profit_Loss'] = formatted_results['Realized_Profit_Loss'].map('${:,.2f}'.format)
        formatted_results['Unrealized_Profit_Loss'] = formatted_results['Unrealized_Profit_Loss'].map('${:,.2f}'.format)
        
        # Display the table
        print(f"\nTrading Results for Period: {period}\n")
        print(tabulate(formatted_results, headers='keys', tablefmt='grid', showindex=False))

        # Calculate totals
        total_realized = filtered_results['Realized_Profit_Loss'].sum()
        total_unrealized = filtered_results['Unrealized_Profit_Loss'].sum()
        
        # Display totals
        print(f"\nTotal Realized Profit/Loss: ${total_realized:,.2f}")
        print(f"Total Unrealized Profit/Loss: ${total_unrealized:,.2f}")
        print(f"Total P&L: ${total_realized + total_unrealized:,.2f}")
        
        # Return the results for further processing if needed
        return grouped
    
    def visualize_results(self, period='All'):
        """Create a visualization of the results"""
        if self.results is None:
            self.combine_results()
        
        filtered_results = self.filter_by_period(period)
        
        if filtered_results.empty:
            print(f"No transactions found for the period: {period}")
            return
        
        # Group by Symbol and Transaction_Type
        grouped = filtered_results.groupby(['Symbol', 'Transaction_Type']).agg({
            'Realized_Profit_Loss': 'sum',
            'Unrealized_Profit_Loss': 'sum'
        }).reset_index()
        
        # Create a stacked bar chart
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        symbols = []
        transaction_types = []
        realized_values = []
        unrealized_values = []
        
        for _, row in grouped.iterrows():
            symbol_type = f"{row['Symbol']} ({row['Transaction_Type']})"
            symbols.append(symbol_type)
            transaction_types.append(row['Transaction_Type'])
            realized_values.append(row['Realized_Profit_Loss'])
            unrealized_values.append(row['Unrealized_Profit_Loss'])
        
        # Create indices for bars
        x = np.arange(len(symbols))
        width = 0.35
        
        # Create bars
        plt.bar(x, realized_values, width, label='Realized P&L')
        plt.bar(x, unrealized_values, width, bottom=realized_values, label='Unrealized P&L')
        
        # Add labels and title
        plt.xlabel('Symbol (Transaction Type)')
        plt.ylabel('Profit/Loss ($)')
        plt.title(f'Profit/Loss by Symbol and Transaction Type - {period}')
        plt.xticks(x, symbols, rotation=45, ha='right')
        plt.legend()
        
        # Add values on top of bars
        for i, (realized, unrealized) in enumerate(zip(realized_values, unrealized_values)):
            plt.text(i, realized/2, f"${realized:,.0f}", ha='center', va='center')
            if unrealized != 0:
                plt.text(i, realized + unrealized/2, f"${unrealized:,.0f}", ha='center', va='center')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create the analysis object
    analysis = TradingAnalysis()
    
    # Load data
    file_path = 'A_Robinhood_Tx_2025.csv'
    if not analysis.load_data(file_path):
        print("Failed to load data. Exiting.")
        return
    
    # Categorize transactions
    analysis.categorize_transactions()
    
    # Calculate profit/loss
    analysis.combine_results()
    
    # Display results for different periods
    periods = ['Current Month', 'Last Month', 'Year-to-date', 'All']
    
    for period in periods:
        analysis.display_results(period)
        print("\n" + "-"*50 + "\n")
    
    # Visualize results for the 'All' period
    analysis.visualize_results('All')

if __name__ == "__main__":
    main()