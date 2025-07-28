import re
import urllib.parse
from bs4 import BeautifulSoup
import requests

class URLFeatureExtractor:
    def __init__(self):
        self.features = {}
    
    def extract_url_features(self, url):
        """Extract features from URL"""
        parsed = urllib.parse.urlparse(url)
        
        features = {
            'url_length': len(url),
            'domain_length': len(parsed.netloc),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_digits': sum(c.isdigit() for c in url),
            'has_https': int(parsed.scheme == 'https'),
            'has_ip': self._has_ip(parsed.netloc),
        }
        
        return features
    
    def _has_ip(self, domain):
        """Check if domain contains IP address"""
        pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return int(bool(re.match(pattern, domain)))