import MetaTrader5 as mt5

if not mt5.initialize():
    print("MT5 initialize() failed")
    print("Error code:", mt5.last_error())
else:
    print("MT5 connected successfully")
    account_info = mt5.account_info()
    print("Account Info:", account_info)
    mt5.shutdown()