from google.cloud import storage
from google.cloud import pubsub_v1
from pathlib import Path
from concurrent import futures
import os
import json
import re
import glob


def get_bucket_object_name(url: str):
    matches = re.match("gs://(.*?)/(.*)", url)
    if matches:
        bucket, object_name = matches.groups()
    else:
        raise Exception("invalid url pattern")
    return bucket, object_name



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The ID of your GCS object
    # source_blob_name = "storage-object-name"
    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(contents)
    print(
        f"{destination_blob_name} with contents {contents} has been uploaded to {bucket_name}."
    )


def upload_blobs_from_directory(
    directory_path: str, dest_bucket_name: str, dest_blob_name: str
):
    storage_client = storage.Client()
    rel_paths = glob.glob(directory_path + "/**", recursive=True)
    bucket = storage_client.bucket(dest_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[2:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

    print(
        f"Folder: {directory_path} has been uploaded to {dest_bucket_name}/{dest_blob_name}."
    )


def download_file_from_gcp(bucket_name, blob_name, dst_folder, dst_name: str):
    output_dir = Path(dst_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_file_name = f"{dst_folder}/{dst_name}"
    download_blob(bucket_name, blob_name, destination_file_name)

    return destination_file_name

def publish_message(project_id, topic_id, data_str, job_id, job_status):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    data = data_str.encode("utf-8")
    future = publisher.publish(
        topic_path, data, origin="tvm-job-vm", job_id=str(job_id), job_status=job_status
    )
    print(future.result())

