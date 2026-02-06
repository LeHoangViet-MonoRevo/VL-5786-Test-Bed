import functools
import io
import json
import os
from urllib.parse import urlparse

import boto3
import requests

from config import settings
from constants import constants


class S3Public:
    def __int__(self):
        pass

    def download_public_files_from_s3(self, link_s3_public):
        response = requests.get(link_s3_public)
        return response.content


s3_public = S3Public()


class S3:
    def __int__(self):
        pass

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_s3_resource():
        s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            aws_session_token=None,
            region_name="ap-northeast-1",
            config=boto3.session.Config(signature_version="s3v4"),
            verify=False,
        )
        return s3_resource

    @staticmethod
    def load_s3_resource_from_credentials(credentials: dict):
        s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=constants.REGION,
        )
        return s3_resource

    @staticmethod
    def get_aws_credentials(idtoken):
        client = boto3.client("cognito-identity", region_name=constants.REGION)
        identity_response = client.get_id(
            IdentityPoolId=settings.AWS_COGNITO_ID_POOL_ID,
            Logins={
                f"cognito-idp.{constants.REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}": idtoken
            },
        )
        identity_id = identity_response["IdentityId"]
        credentials_response = client.get_credentials_for_identity(
            IdentityId=identity_id,
            Logins={
                f"cognito-idp.{constants.REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}": idtoken
            },
        )

        return credentials_response["Credentials"]

    def download_s3_indexer(self, bucket_name, s3_folder, local_dir=None):
        bucket = self.load_s3_resource().Bucket(bucket_name)
        bucket_object = bucket.objects.filter(Prefix=s3_folder + "/")
        if len(list(bucket_object)) != 0:
            for obj in bucket_object:
                if (
                    obj.key == f"{s3_folder}/.DS_Store"
                    or obj.key == f"{s3_folder}/metadata-files/.DS_Store"
                ):
                    continue
                target = (
                    obj.key
                    if local_dir is None
                    else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
                )
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                bucket.download_file(obj.key, target)
            return True
        else:
            return False

    def list_files_from_folder(self, path=None, bucket_name=None, folder=None):
        if path is not None:
            s3_parser = urlparse(path, allow_fragments=False)
            bucket_name = s3_parser.netloc
            folder = s3_parser.path
        bucket = self.load_s3_resource().Bucket(bucket_name)
        files = []
        for f in bucket.objects.filter(Prefix=folder[1:]):
            files.append(f)
        return files

    def download_files_to_local(
        self,
        bucket_name=None,
        file_path=None,
        remote_location=None,
        local_dir=None,
        s3_resource=None,
    ):
        if remote_location is not None:
            s3_parser = urlparse(remote_location, allow_fragments=False)
            bucket_name = s3_parser.netloc
            file_path = s3_parser.path
        if s3_resource is None:
            s3_resource = self.load_s3_resource()
        bucket = s3_resource.Bucket(bucket_name)
        bucket.download_file(file_path, local_dir)

    def download_image_to_variable(
        self,
        s3_resource,
        bucket_name,
        file_path=None,
        remote_location=None,
        local_dir=None,
    ):
        if remote_location is not None:
            s3_parser = urlparse(remote_location, allow_fragments=False)
            bucket_name = s3_parser.netloc
            file_path = s3_parser.path.lstrip("/")

        byte_stream = io.BytesIO()
        bucket = s3_resource.Bucket(bucket_name)
        bucket.download_fileobj(file_path, byte_stream)
        byte_stream.seek(0)

        return byte_stream.getvalue()

    def download_folder_to_local(
        self, bucket_name=None, folder_path=None, remote_location=None, local_dir=None
    ):
        if remote_location is not None:
            s3_parser = urlparse(remote_location, allow_fragments=False)
            bucket_name = s3_parser.netloc
            folder_path = s3_parser.path.lstrip("/")

        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        bucket = self.load_s3_resource().Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=folder_path):
            local_file_path = os.path.join(
                local_dir, os.path.relpath(obj.key, folder_path)
            )
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            bucket.download_file(obj.key, local_file_path)

    def upload_file(
        self, bucket_name=None, s3_key=None, file_path=None, s3_resource=None
    ):
        if s3_resource is None:
            s3_resource = self.load_s3_resource()
        bucket = s3_resource.Bucket(bucket_name)
        bucket.upload_file(file_path, s3_key)

    def put_object(self, input: list, bucket_name=None, s3_key=None, s3_resource=None):
        if s3_resource is None:
            s3_resource = self.load_s3_resource()
        s3_resource.Bucket(bucket_name).put_object(
            Key=s3_key,
            Body=json.dumps(input, ensure_ascii=False, indent=4).encode("utf-8"),
            ContentType="application/json",
        )


class S3Public:
    def __int__(self):
        pass

    def download_public_files(self, link_s3_public):
        response = requests.get(link_s3_public)
        return response.content


s3 = S3()
