import unittest
from app import app  # Đảm bảo app.py nằm trong cùng thư mục hoặc được import đúng cách


class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        # Tạo một client thử nghiệm cho Flask app
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        """
        Kiểm tra rằng trang chủ ('/') trả về HTTP 200.
        """
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome", response.data)  # Kiểm tra nội dung (tuỳ chỉnh theo trang chủ)

    def test_prediction(self):
        """
        Kiểm tra API /predict trả về dự đoán với dữ liệu đầu vào.
        """
        features = {"features": [1.2, -0.4, 3.1, 0.6, -0.5, 1.0, 2.1, -1.0, 0.7, -0.3]}
        response = self.app.post('/predict', json=features)

        # Kiểm tra trạng thái HTTP
        self.assertEqual(response.status_code, 200)

        # Đảm bảo phản hồi có trường 'prediction'
        self.assertIn('prediction', response.json)

        # Kiểm tra giá trị prediction là số nguyên hoặc hợp lệ
        prediction = response.json.get('prediction')
        self.assertIsInstance(prediction, int)  # Giả định prediction là số nguyên


if __name__ == "__main__":
    unittest.main()
