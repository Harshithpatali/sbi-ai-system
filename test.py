import torch
import transformers
import yfinance as yf
import pandas as pd

print("PyTorch:", torch.__version__)
print("Transformers Imported")
print("Yahoo Finance Working")

data = yf.download("SBIN.NS", period="5d")

print(data.head())