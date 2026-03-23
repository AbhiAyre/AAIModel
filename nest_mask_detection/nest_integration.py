"""Google Nest integration utilities."""
import logging
import base64
import json
from typing import Optional, Dict, Any
import requests
from google.cloud import pubsub_v1
from google.cloud import storage

logger = logging.getLogger(__name__)


class NestDeviceAccess:
    """Google Nest Device Access API client."""

    def __init__(self, project_id: str, api_key: str, device_id: str):
        """
        Initialize Nest Device Access client.

        Args:
            project_id: Google Cloud project ID
            api_key: Google Home API key
            device_id: Nest device ID
        """
        self.project_id = project_id
        self.api_key = api_key
        self.device_id = device_id
        self.base_url = (
            f"https://smartdevicemanagement.googleapis.com/v1/"
            f"enterprises/{project_id}/devices/{device_id}"
        )

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.

        Returns:
            Device info dictionary
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(self.base_url, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_event_image(self, event_image_url: str) -> Optional[bytes]:
        """
        Download event image from Nest.

        Args:
            event_image_url: URL from Pub/Sub event

        Returns:
            Image bytes or None
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                event_image_url,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download event image: {e}")
            return None

    def get_live_stream_token(self) -> Optional[str]:
        """
        Get live stream token for RTSP/WebRTC.

        Returns:
            Stream token or None
        """
        try:
            url = f"{self.base_url}:executeCommand"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "command": "sdm.devices.commands.CameraLiveStream.GenerateRtspStream",
                "params": {}
            }
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", {}).get("streamToken")
        except Exception as e:
            logger.error(f"Failed to get live stream token: {e}")
            return None


class NestPubSubListener:
    """Listen to Nest events from Google Pub/Sub."""

    def __init__(
        self,
        project_id: str,
        subscription_id: str,
        callback,
    ):
        """
        Initialize Pub/Sub listener.

        Args:
            project_id: Google Cloud project ID
            subscription_id: Pub/Sub subscription ID
            callback: Callback function for each event
        """
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.callback = callback
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(
            project_id, subscription_id
        )
        logger.info(
            f"Initialized Pub/Sub listener for {subscription_id}"
        )

    def message_callback(self, message):
        """
        Handle incoming Pub/Sub message.

        Args:
            message: Pub/Sub message
        """
        try:
            # Decode message
            payload = json.loads(message.data.decode("utf-8"))
            logger.info(f"Received event: {json.dumps(payload, indent=2)}")

            # Parse event
            event_data = self._parse_event(payload)
            if event_data:
                self.callback(event_data)

            # Acknowledge message
            message.ack()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            message.ack()  # Ack even on error to avoid redelivery

    def _parse_event(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse Nest event payload.

        Args:
            payload: Pub/Sub message payload

        Returns:
            Parsed event or None
        """
        try:
            # Expected structure from Nest Device Access
            events = payload.get("resourceUpdate", {}).get("events", [])

            for event in events:
                if "image" in event:
                    image_event = event["image"]
                    return {
                        "event_type": "image",
                        "event_id": image_event.get("eventId"),
                        "timestamp": image_event.get("timestamp"),
                        "url": image_event.get("url"),
                    }

            return None
        except Exception as e:
            logger.error(f"Error parsing event: {e}")
            return None

    def start_listening(self):
        """Start listening for events (blocking)."""
        logger.info("Starting Pub/Sub listener...")
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path, callback=self.message_callback
        )

        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("Stopping listener...")
            streaming_pull_future.cancel()

    def stop_listening(self):
        """Stop listening for events."""
        logger.info("Stopping listener...")


class NestCloudStorage:
    """Google Cloud Storage utilities for storing predictions."""

    def __init__(self, project_id: str, bucket_name: str):
        """
        Initialize Cloud Storage client.

        Args:
            project_id: Google Cloud project ID
            bucket_name: GCS bucket name
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)

    def upload_image(self, local_path: str, remote_path: str) -> bool:
        """
        Upload image to Cloud Storage.

        Args:
            local_path: Local image path
            remote_path: Remote blob path

        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return False

    def download_image(self, remote_path: str, local_path: str) -> bool:
        """
        Download image from Cloud Storage.

        Args:
            remote_path: Remote blob path
            local_path: Local image path

        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return False
