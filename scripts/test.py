from forecasting_tools.util.file_manipulation import write_json_file

if __name__ == "__main__":
    data = write_json_file("logs/claims.json", [{"a": 1}, {"b": 2}])
