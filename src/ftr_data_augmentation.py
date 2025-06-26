import random
import datetime
from decimal import Decimal
import uuid

# Configuration
START_DATE = datetime.datetime(2019, 1, 1)
END_DATE = datetime.datetime(2025, 5, 31)
RECORDS_PER_MONTH = 10000  # Approximately 60,000 total records

# Sample data
BANKS = [
    ("CITI001", "Citibank N.A.", "CITIUS33"),
    ("JPM001", "JPMorgan Chase Bank", "CHASUS33"),
    ("BAC001", "Bank of America", "BOFAUS3N"),
    ("WFC001", "Wells Fargo Bank", "WFBIUS6S"),
    ("USB001", "U.S. Bank", "USBKUS44"),
    ("PNC001", "PNC Bank", "PNCCUS33"),
    ("TD001", "TD Bank", "NRTHUS33"),
    ("HSBC001", "HSBC Bank USA", "MRMDUS33")
]

CUSTOMER_NAMES = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "David Wilson",
    "Lisa Anderson", "Robert Taylor", "Jennifer Martinez", "Christopher Garcia", "Amanda Rodriguez",
    "Matthew Lopez", "Jessica Hernandez", "Daniel Gonzalez", "Ashley Perez", "James Thompson",
    "Samantha White", "Kevin Lee", "Rachel Harris", "Brian Clark", "Nicole Lewis",
    "ABC Corporation", "XYZ Industries", "Global Tech Solutions", "Metro Construction Co",
    "Premier Medical Group", "Sunshine Retail LLC", "Mountain View Consulting", "Riverside Manufacturing"
]

TRANSACTION_TYPES = ["CREDIT", "DEBIT"]
TRANSACTION_CHANNELS = ["ONLINE", "ATM", "BRANCH", "MOBILE", "WIRE", "ACH", "CHECK"]
ACCOUNT_TYPES = ["SAVINGS", "CHECKING", "BUSINESS", "LOAN", "CREDIT_CARD", "INVESTMENT"]
CUSTOMER_TYPES = ["INDIVIDUAL", "CORPORATE", "GOVERNMENT", "NON_PROFIT"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "JPY"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
STATES = ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "OH"]
COUNTRIES = ["USA", "Canada", "Mexico", "UK", "Germany", "France", "Japan", "Australia"]

def generate_random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + datetime.timedelta(days=random_days)

def generate_transaction_data():
    records = []
    
    for i in range(60000):  # 5 years of data
        # Generate base transaction info
        trans_date = generate_random_date(START_DATE, END_DATE)
        posting_date = trans_date + datetime.timedelta(days=random.randint(0, 2))
        value_date = posting_date + datetime.timedelta(days=random.randint(0, 1))
        
        # Our bank info
        our_bank = random.choice(BANKS)
        our_customer_name = random.choice(CUSTOMER_NAMES)
        our_customer_type = "CORPORATE" if "Corp" in our_customer_name or "LLC" in our_customer_name or "Inc" in our_customer_name else "INDIVIDUAL"
        
        # Counterparty info
        counterparty_bank = random.choice(BANKS)
        counterparty_customer_name = random.choice(CUSTOMER_NAMES)
        
        # Transaction amount
        amount = round(random.uniform(10.00, 50000.00), 2)
        
        record = {
            'transaction_id': f"TXN{trans_date.strftime('%Y%m%d')}{str(uuid.uuid4())[:8].upper()}",
            'transaction_reference': f"REF{random.randint(100000, 999999)}",
            'transaction_date': trans_date.strftime('%Y-%m-%d %H:%M:%S'),
            'posting_date': posting_date.strftime('%Y-%m-%d %H:%M:%S'),
            'value_date': value_date.strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_type': random.choice(TRANSACTION_TYPES),
            'transaction_amount': amount,
            'transaction_currency': random.choice(CURRENCIES),
            'transaction_status': random.choices(['COMPLETED', 'PENDING', 'FAILED'], weights=[85, 10, 5])[0],
            'transaction_channel': random.choice(TRANSACTION_CHANNELS),
            'transaction_description': f"Payment for services - {random.randint(1000, 9999)}",
            
            # Our bank
            'our_bank_code': our_bank[0],
            'our_bank_name': our_bank[1],
            'our_branch_code': f"BR{random.randint(100, 999)}",
            'our_branch_name': f"{random.choice(CITIES)} Branch",
            
            # Our customer
            'our_customer_id': f"CUST{random.randint(100000, 999999)}",
            'our_customer_name': our_customer_name,
            'our_customer_type': our_customer_type,
            'our_account_number': f"{random.randint(1000000000, 9999999999)}",
            'our_account_type': random.choice(ACCOUNT_TYPES),
            'our_customer_email': f"{our_customer_name.lower().replace(' ', '.')}@email.com",
            'our_customer_phone': f"+1{random.randint(2000000000, 9999999999)}",
            'our_customer_address': f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Park Blvd', 'First St'])}",
            'our_customer_city': random.choice(CITIES),
            'our_customer_state': random.choice(STATES),
            'our_customer_country': random.choice(COUNTRIES),
            'our_customer_postal_code': f"{random.randint(10000, 99999)}",
            
            # Counterparty bank
            'counterparty_bank_code': counterparty_bank[0],
            'counterparty_bank_name': counterparty_bank[1],
            'counterparty_swift_code': counterparty_bank[2],
            'counterparty_routing_number': f"{random.randint(100000000, 999999999)}",
            'counterparty_branch_code': f"BR{random.randint(100, 999)}",
            'counterparty_branch_name': f"{random.choice(CITIES)} Branch",
            
            # Counterparty customer
            'counterparty_customer_id': f"EXT{random.randint(100000, 999999)}",
            'counterparty_customer_name': counterparty_customer_name,
            'counterparty_account_number': f"{random.randint(1000000000, 9999999999)}",
            'counterparty_account_type': random.choice(ACCOUNT_TYPES),
            'counterparty_customer_email': f"{counterparty_customer_name.lower().replace(' ', '.')}@email.com",
            'counterparty_customer_phone': f"+1{random.randint(2000000000, 9999999999)}",
            'counterparty_customer_address': f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Park Blvd', 'First St'])}",
            'counterparty_customer_city': random.choice(CITIES),
            'counterparty_customer_state': random.choice(STATES),
            'counterparty_customer_country': random.choice(COUNTRIES),
            'counterparty_customer_postal_code': f"{random.randint(10000, 99999)}",
            
            # Balance info
            'opening_balance': round(random.uniform(1000.00, 100000.00), 2),
            'closing_balance': round(random.uniform(1000.00, 100000.00), 2),
            'available_balance': round(random.uniform(1000.00, 100000.00), 2),
            
            # Fee info
            'transaction_fee': round(random.uniform(0.00, 25.00), 2),
            'fee_currency': 'USD',
            
            # Additional fields
            'exchange_rate': round(random.uniform(0.5, 2.0), 6),
            'purpose_code': f"P{random.randint(100, 999)}",
            'purpose_description': random.choice(['Business Payment', 'Personal Transfer', 'Salary Payment', 'Investment', 'Loan Payment']),
            'batch_id': f"BATCH{trans_date.strftime('%Y%m%d')}{random.randint(100, 999)}",
            'settlement_datetime': (value_date + datetime.timedelta(hours=random.randint(1, 24))).strftime('%Y-%m-%d %H:%M:%S'),
            'record_source': random.choice(['CORE_BANKING', 'MOBILE_APP', 'ONLINE_BANKING', 'ATM_NETWORK'])
        }
        
        records.append(record)
    
    return records

# Generate data and create SQL file
print("Generating sample data...")
data = generate_transaction_data()

print("Creating SQL insert file...")
with open('financial_transactions_data.sql', 'w') as f:
    f.write("USE dpl_uae_compliance;\n\n")
    f.write("-- Disable foreign key checks and autocommit for faster inserts\n")
    f.write("SET FOREIGN_KEY_CHECKS = 0;\n")
    f.write("SET AUTOCOMMIT = 0;\n\n")
    
    # Insert in batches of 1000
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        f.write(f"-- Batch {i//batch_size + 1}\n")
        f.write("INSERT INTO comp_financial_tran_repos_dly (\n")
        f.write("    transaction_id, transaction_reference, transaction_date, posting_date, value_date,\n")
        f.write("    transaction_type, transaction_amount, transaction_currency, transaction_status, transaction_channel,\n")
        f.write("    transaction_description, our_bank_code, our_bank_name, our_branch_code, our_branch_name,\n")
        f.write("    our_customer_id, our_customer_name, our_customer_type, our_account_number, our_account_type,\n")
        f.write("    our_customer_email, our_customer_phone, our_customer_address, our_customer_city, our_customer_state,\n")
        f.write("    our_customer_country, our_customer_postal_code, counterparty_bank_code, counterparty_bank_name,\n")
        f.write("    counterparty_swift_code, counterparty_routing_number, counterparty_branch_code, counterparty_branch_name,\n")
        f.write("    counterparty_customer_id, counterparty_customer_name, counterparty_account_number, counterparty_account_type,\n")
        f.write("    counterparty_customer_email, counterparty_customer_phone, counterparty_customer_address,\n")
        f.write("    counterparty_customer_city, counterparty_customer_state, counterparty_customer_country,\n")
        f.write("    counterparty_customer_postal_code, opening_balance, closing_balance, available_balance,\n")
        f.write("    transaction_fee, fee_currency, exchange_rate, purpose_code, purpose_description,\n")
        f.write("    batch_id, settlement_datetime, record_source\n")
        f.write(") VALUES\n")
        
        for j, record in enumerate(batch):
            f.write("(")
            values = []
            for key in ['transaction_id', 'transaction_reference', 'transaction_date', 'posting_date', 'value_date',
                       'transaction_type', 'transaction_amount', 'transaction_currency', 'transaction_status', 'transaction_channel',
                       'transaction_description', 'our_bank_code', 'our_bank_name', 'our_branch_code', 'our_branch_name',
                       'our_customer_id', 'our_customer_name', 'our_customer_type', 'our_account_number', 'our_account_type',
                       'our_customer_email', 'our_customer_phone', 'our_customer_address', 'our_customer_city', 'our_customer_state',
                       'our_customer_country', 'our_customer_postal_code', 'counterparty_bank_code', 'counterparty_bank_name',
                       'counterparty_swift_code', 'counterparty_routing_number', 'counterparty_branch_code', 'counterparty_branch_name',
                       'counterparty_customer_id', 'counterparty_customer_name', 'counterparty_account_number', 'counterparty_account_type',
                       'counterparty_customer_email', 'counterparty_customer_phone', 'counterparty_customer_address',
                       'counterparty_customer_city', 'counterparty_customer_state', 'counterparty_customer_country',
                       'counterparty_customer_postal_code', 'opening_balance', 'closing_balance', 'available_balance',
                       'transaction_fee', 'fee_currency', 'exchange_rate', 'purpose_code', 'purpose_description',
                       'batch_id', 'settlement_datetime', 'record_source']:
                
                value = record[key]
                if isinstance(value, str):
                    values.append(f'\'{value.replace("\'", "\\\'")}\' ')   #f"'{value.replace(\"'\", \"\\'\")}'")
                else:
                    values.append(str(value))
            
            f.write(", ".join(values))
            f.write(")")
            
            if j < len(batch) - 1:
                f.write(",\n")
            else:
                f.write(";\n\n")
        
        if (i + batch_size) % 10000 == 0:
            f.write("COMMIT;\n\n")
    
    f.write("-- Final commit and restore settings\n")
    f.write("COMMIT;\n")
    f.write("SET FOREIGN_KEY_CHECKS = 1;\n")
    f.write("SET AUTOCOMMIT = 1;\n")

print("SQL file 'financial_transactions_data.sql' has been created successfully!")
print(f"Generated {len(data)} records spanning from {START_DATE.date()} to {END_DATE.date()}")