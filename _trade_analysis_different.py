def extract_option_data(description):
    """Extract option details from Description """
    if not isinstance(description, str):
        return None
    
    # Pattern 1: Standard format (PUT C3 AI INC $21.5 EXP 09/20/24)
    pattern1 = r"(CALL|PUT)\s+(.*?)\s+\$([\d.]+)\s+EXP\s+(\d{2}/\d{2}/\d{2})"
    # Pattern 2: Alternate format (LCID 1/31/2025 Put $2.50)
    pattern2 = r"(.*?)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(Call|Put)\s+\$([\d.]+)"
    
    match = re.search(pattern1, description) or re.search(pattern2, description)
    if not match:
        return None
    
    if len(match.groups()) == 4:  # Standard format
        option_type, underlying, strike, expiry = match.groups()
    else:  # Alternate format
        underlying, expiry, option_type, strike = match.groups()
        option_type = option_type.upper()  # Convert "Put" to "PUT"
    
    return {
        'option_type': option_type,
        'underlying': underlying.strip(),
        'strike': float(strike),
        'expiry': expiry
    }

def match_option_trades(df, date_extent):
    """Match buy/sell/expire transactions for options with partial quantity matching"""
    # ... [previous code until matches = []]
    
    # Process each option transaction
    for idx, row in options_df.iterrows():
        if idx in processed_indices:
            continue

        option_id = row['Description']
        action = row['Action']
        quantity = row['Quantity']
        
        # Determine matching actions
        if action == 'Buy to Open':
            matching_actions = ['Sell to Close', 'Expired']
        elif action == 'Sell to Open':
            matching_actions = ['Buy to Close', 'Expired']
        else:
            continue  # Skip close actions as they'll be matched from open actions

        # Find all potential matches (same description and matching action)
        potential_matches = options_df[
            (options_df['Description'] == option_id) & 
            (~options_df.index.isin(processed_indices)) & 
            (options_df['Action'].isin(matching_actions))
        ].sort_values('Date')
        
        remaining_qty = quantity
        matched_qty = 0
        
        # Try to match as much quantity as possible
        for _, match_row in potential_matches.iterrows():
            if remaining_qty <= 0:
                break
                
            match_qty = min(remaining_qty, match_row['Quantity'])
            if match_qty <= 0:
                continue
                
            # Calculate P&L for matched portion
            if action == 'Buy to Open':
                buy_amount = (row['Amount'] / row['Quantity']) * match_qty
                sell_amount = (match_row['Amount'] / match_row['Quantity']) * match_qty
                profit_loss = sell_amount + buy_amount
                position_type = 'Long'
            else:  # Sell to Open
                sell_amount = (row['Amount'] / row['Quantity']) * match_qty
                buy_amount = (match_row['Amount'] / match_row['Quantity']) * match_qty
                profit_loss = buy_amount + sell_amount
                position_type = 'Short'
            
            # Calculate hold days
            hold_days = (match_row['Date'] - row['Date']).days
            
            # Extract option details
            option_details = extract_option_data(row['Description'])
            
            matches.append({
                'Symbol': row['Symbol'],
                'Underlying': option_details['underlying'] if option_details else '',
                'Option Type': option_details['option_type'] if option_details else '',
                'Strike': option_details['strike'] if option_details else '',
                'Expiry': option_details['expiry'] if option_details else '',
                'Position': position_type,
                'Quantity': match_qty,
                'Open Date': row['Date'],
                'Close Date': match_row['Date'],
                'Open Price': row['Price'],
                'Close Price': match_row['Price'],
                'Hold Days': hold_days,
                'P&L': profit_loss,
                'Tx_Month': match_row['Date'].month,
                'Matched_Qty': match_qty,
                'Unmatched_Qty': 0  # Will update remaining below
            })
            
            # Update quantities
            remaining_qty -= match_qty
            matched_qty += match_qty
            
            # Mark the matched portion as processed
            if match_row['Quantity'] == match_qty:
                processed_indices.add(match_row.name)
            else:
                # Update the remaining quantity in the match row
                options_df.at[match_row.name, 'Quantity'] -= match_qty
                
        # If there's remaining quantity, record as unmatched
        if remaining_qty > 0:
            option_details = extract_option_data(row['Description'])
            matches.append({
                'Symbol': row['Symbol'],
                'Underlying': option_details['underlying'] if option_details else '',
                'Option Type': option_details['option_type'] if option_details else '',
                'Strike': option_details['strike'] if option_details else '',
                'Expiry': option_details['expiry'] if option_details else '',
                'Position': 'Long' if action == 'Buy to Open' else 'Short',
                'Quantity': remaining_qty,
                'Open Date': row['Date'],
                'Close Date': pd.NaT,
                'Open Price': row['Price'],
                'Close Price': 0,
                'Hold Days': 0,
                'P&L': 0,
                'Tx_Month': row['Date'].month,
                'Matched_Qty': matched_qty,
                'Unmatched_Qty': remaining_qty
            })
            
        processed_indices.add(idx)
    
    # ... [rest of the function remains the same]