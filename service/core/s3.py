import boto3
import tempfile
import os
from contextlib import contextmanager
from botocore.exceptions import NoCredentialsError

@contextmanager
def download_temp_file(bucket_name: str, object_name: str):
    s3 = boto3.client("s3")

    suffix = os.path.basename(object_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{suffix}")
    filepath = temp_file.name

    try:
        s3.download_fileobj(bucket_name, object_name, temp_file)
        temp_file.close()
        yield filepath
    except NoCredentialsError:
        print("Credentials not available")
        raise
    except Exception as e:
        print(e)
        raise
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)