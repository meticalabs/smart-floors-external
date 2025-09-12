import logging
from typing import Dict, List, Optional

import requests


class ApplovinManagementApiClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Api-Key": f"{api_key}",
        }

    def get_ad_units(self, fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch all ad units from AppLovin API

        Args:
            fields: Optional list of additional fields to include in the response.
                   Possible values include 'ad_network_settings', 'frequency_capping_settings',
                   and 'bid_floors'.
        """
        try:
            url = f"{self.base_url}/ad_units"
            params = {}

            if fields:
                params["fields"] = ",".join(fields)

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ad units: {str(e)}")
            raise

    def get_ad_unit(self, ad_unit_id: str, fields: Optional[List[str]] = None) -> Dict:
        """
        Fetch a specific ad unit from AppLovin API by its ID

        Args:
            ad_unit_id: The ID of the ad unit to fetch
            fields: Optional list of additional fields to include in the response.
                   Possible values include 'ad_network_settings', 'frequency_capping_settings',
                   and 'bid_floors'.

        Returns:
            Dict containing the ad unit information
        """
        try:
            url = f"{self.base_url}/ad_unit/{ad_unit_id}"
            params = {}

            if fields:
                params["fields"] = ",".join(fields)

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ad unit {ad_unit_id}: {str(e)}")
            raise

    def get_ad_unit_experiment(self, ad_unit_id: str, fields: Optional[List[str]] = None) -> Dict:
        """
        Fetch details about an ad unit experiment from AppLovin API

        Args:
            ad_unit_id: The ID of the ad unit whose experiment to fetch
            fields: Optional list of additional fields to include in the response.
                   Possible values include 'ad_network_settings', 'disabled_ad_network_settings',
                   'frequency_capping_settings', 'bid_floors', 'segments', 'banner_refresh_settings',
                   and 'mrec_refresh_settings'.

        Returns:
            Dict containing the ad unit experiment information
        """
        try:
            url = f"{self.base_url}/ad_unit_experiment/{ad_unit_id}"
            params = {}

            if fields:
                params["fields"] = ",".join(fields)

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching ad unit experiment for ad unit {ad_unit_id}: {str(e)}")
            raise

    def update_ad_unit(self, ad_unit_id: str, ad_unit_data: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Update an ad unit via AppLovin API

        Args:
            ad_unit_id: The ID of the ad unit to update
            ad_unit_data: The complete ad unit object (if provided, kwargs will update specific fields)
            **kwargs: Additional fields to update. Common fields include:
                     - bid_floors: List of bid floor configurations
                     - ad_network_settings: List of ad network configurations
                     - frequency_capping_settings: List of frequency capping configurations
                     - disabled: Boolean to enable/disable the ad unit
                     - name: New name for the ad unit

        Returns:
            Dict containing the API response
        """
        try:
            url = f"{self.base_url}/ad_unit/{ad_unit_id}"

            # If complete ad unit data is provided, use it as base and update with kwargs
            if ad_unit_data:
                payload = ad_unit_data.copy()
                # Update specific fields from kwargs
                for key, value in kwargs.items():
                    if value is not None:
                        payload[key] = value
            else:
                # Fallback to old behavior - get the ad unit first, then update
                current_ad_unit = self.get_ad_unit(
                    ad_unit_id, fields=["ad_network_settings", "frequency_capping_settings", "bid_floors"]
                )
                payload = current_ad_unit.copy()
                # Update specific fields from kwargs
                for key, value in kwargs.items():
                    if value is not None:
                        payload[key] = value

            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error updating ad unit {ad_unit_id}: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    logging.error(f"API Error Details: {error_details}")
                except Exception as e:
                    logging.error(f"API Error Response: {e.response.text}")
            raise
