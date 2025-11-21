import requests

def download_file_from_presigned_url(url, save_path):
    if not url:
        print(">>> [Error] Invalid URL.")
        return

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

    except requests.exceptions.HTTPError as http_err:
        print(f">>>[Error] {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f">>>[Error] {conn_err}")
    except requests.exceptions.RequestException as req_err:
        print(f">>>[Error] {req_err}")
    except Exception as e:
        print(f">>>[Error] {e}")