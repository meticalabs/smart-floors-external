import logging
from typing import Dict, List, Optional

import requests


class MeticaManagementApiClient:
    def __init__(self, customer_id: str, app_id: str, base_url: str = "https://services-alb.prod-eu.metica.com/management/api/v1"):
        self.customer_id = customer_id
        self.app_id = app_id
        self.base_url = base_url
        self.headers = {
            "accept": "*/*",
        }

    def get_ad_units(self, fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch all ad units from Metica API

        Args:
            fields: Optional list of additional fields to include in the response.
                   This parameter is included for compatibility but may not be used by Metica API.

        Returns:
            List of ad unit dictionaries
        """
        try:
            # The correct base URL for this endpoint does NOT include '/monitor'
            url = f"{self.base_url}/customer/{self.customer_id}/application/{self.app_id}/adUnit"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ad units: {str(e)}")
            raise

    def get_ad_unit(self, ad_unit_id: str, fields: Optional[List[str]] = None) -> Dict:
        """
        Fetch a specific ad unit from Metica API by its ID

        Args:
            ad_unit_id: The ID of the ad unit to fetch
            fields: Optional list of additional fields to include in the response.
                   This parameter is included for compatibility but may not be used by Metica API.

        Returns:
            Dict containing the ad unit information
        """
        try:
            url = f"{self.base_url}/customer/{self.customer_id}/application/{self.app_id}/adUnit/{ad_unit_id}"
            params = {}

            if fields:
                params["fields"] = ",".join(fields)

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ad unit {ad_unit_id}: {str(e)}")
            raise

    def create_ad_unit(self, ad_unit_data: Dict) -> Dict:
        """
        Create a new ad unit via Metica API

        Args:
            ad_unit_data: The ad unit data to create

        Returns:
            Dict containing the API response
        """
        try:
            url = f"{self.base_url}/customer/{self.customer_id}/application/{self.app_id}/adUnit"
            
            response = requests.post(url, headers=self.headers, json=ad_unit_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error creating ad unit: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    logging.error(f"API Error Details: {error_details}")
                except Exception:
                    logging.error(f"API Error Response: {e.response.text}")
            raise

    def update_ad_unit(self, ad_unit_id: str, ad_unit_data: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Update an ad unit via Metica API

        Args:
            ad_unit_id: The ID of the ad unit to update
            ad_unit_data: The complete ad unit object (if provided, kwargs will update specific fields)
            **kwargs: Additional fields to update

        Returns:
            Dict containing the API response
        """
        try:
            url = f"{self.base_url}/customer/{self.customer_id}/application/{self.app_id}/adUnit/{ad_unit_id}"

            # If complete ad unit data is provided, use it as base and update with kwargs
            if ad_unit_data:
                payload = ad_unit_data.copy()
                # Update specific fields from kwargs
                for key, value in kwargs.items():
                    if value is not None:
                        payload[key] = value
            else:
                # Fallback to old behavior - get the ad unit first, then update
                current_ad_unit = self.get_ad_unit(ad_unit_id)
                payload = current_ad_unit.copy()
                # Update specific fields from kwargs
                for key, value in kwargs.items():
                    if value is not None:
                        payload[key] = value

            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error updating ad unit {ad_unit_id}: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    logging.error(f"API Error Details: {error_details}")
                except Exception:
                    logging.error(f"API Error Response: {e.response.text}")
            raise

    def delete_ad_unit(self, ad_unit_id: str) -> Dict:
        """
        Delete an ad unit via Metica API

        Args:
            ad_unit_id: The ID of the ad unit to delete

        Returns:
            Dict containing the API response
        """
        try:
            url = f"{self.base_url}/customer/{self.customer_id}/application/{self.app_id}/adUnit/{ad_unit_id}"
            
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error deleting ad unit {ad_unit_id}: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    logging.error(f"API Error Details: {error_details}")
                except Exception:
                    logging.error(f"API Error Response: {e.response.text}")
            raise
