import os

file_path = 'house_price_model.pkl'
if os.path.exists(file_path):
    print(f"{file_path} exists and has size {os.path.getsize(file_path)} bytes")
else:
    print(f"{file_path} does not exist")
