import requests

def download_file_from_presigned_url(url, save_path):
    if not url or url == "YOUR_PRESIGNED_URL_HERE":
        print("오류: PRESIGNED_URL 변수에 유효한 URL을 입력해주세요.")
        return

    print(f"다운로드 시작: {url}")

    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                chunk_size = 8192
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

            print(f"파일 다운로드 완료: {save_path}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 오류 발생: {http_err}")
        print("Presigned URL이 만료되었거나 권한이 없는지 확인해보세요.")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"연결 오류 발생: {conn_err}")
        print("네트워크 연결을 확인해보세요.")
    except requests.exceptions.RequestException as req_err:
        print(f"요청 오류 발생: {req_err}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")