import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open
import responses
from simple_email import send_email, generate_analysis


class TestGenerateAnalysis:
    """Test the generate_analysis function"""
    
    @patch('simple_email.os.getenv')
    @patch('simple_email.genai.configure')
    @patch('simple_email.genai.GenerativeModel')
    @patch('simple_email.convert_chats_to_json')
    def test_generate_analysis_success(self, mock_convert, mock_model_class, mock_configure, mock_getenv):
        """Test successful analysis generation"""
        # Setup mocks
        mock_getenv.return_value = "fake_api_key"
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "Score: 85/100\nAnalysis: Good empathetic responses."
        mock_model_class.return_value = mock_model
        mock_convert.return_value = [{"user": "How are you?", "system": "I'm okay"}]
        
        # Test data
        chats = ["How are you?", "I'm okay", "Really? You seem tired", "Yeah, haven't been sleeping well"]
        
        # Execute
        result = generate_analysis(chats)
        
        # Assert
        assert result == "Score: 85/100\nAnalysis: Good empathetic responses."
        mock_convert.assert_called_once_with(chats, "chats.json")
        mock_configure.assert_called_once_with(api_key="fake_api_key")
        mock_model_class.assert_called_once_with("gemini-2.5-flash")
        mock_model.generate_content.assert_called_once()
    
    @patch('simple_email.os.getenv')
    def test_generate_analysis_missing_api_key(self, mock_getenv):
        """Test analysis when API key is missing"""
        mock_getenv.return_value = None
        
        chats = ["Hello", "Hi there"]
        
        # This should raise an exception or handle gracefully
        with pytest.raises(Exception):
            generate_analysis(chats)
    
    @patch('simple_email.os.getenv')
    @patch('simple_email.genai.configure')
    @patch('simple_email.genai.GenerativeModel')
    @patch('simple_email.convert_chats_to_json')
    def test_generate_analysis_empty_chats(self, mock_convert, mock_model_class, mock_configure, mock_getenv):
        """Test analysis with empty chat list"""
        mock_getenv.return_value = "fake_api_key"
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "No conversation to analyze."
        mock_model_class.return_value = mock_model
        mock_convert.return_value = []
        
        result = generate_analysis([])
        
        assert result == "No conversation to analyze."
        mock_convert.assert_called_once_with([], "chats.json")


class TestSendEmail:
    """Test the send_email function"""
    
    @responses.activate
    @patch('simple_email.generate_analysis')
    def test_send_email_success(self, mock_generate):
        """Test successful email sending"""
        # Setup mock response
        mock_generate.return_value = "Score: 75/100\nUser showed good empathy."
        
        responses.add(
            responses.POST,
            "https://api.resend.com/emails",
            json={"id": "email_123"},
            status=200
        )
        
        # Test data
        chats = ["How are you feeling?", "Not great lately", "What's been bothering you?", "Just stressed about school"]
        
        # Execute
        result = send_email("test@example.com", chats)
        
        # Assert
        assert result is True
        assert len(responses.calls) == 1
        
        # Check request content
        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["to"] == ["test@example.com"]
        assert request_body["subject"] == "Peer Support Analysis Report"
        assert "Score: 75/100" in request_body["html"]
        assert "Research References" in request_body["html"]
        assert "https://doi.org/10.1186/s13034-022-00481-y" in request_body["html"]
        assert "https://doi.org/10.1111/jcap.12299" in request_body["html"]
    
    @responses.activate
    @patch('simple_email.generate_analysis')
    def test_send_email_failure(self, mock_generate):
        """Test email sending failure"""
        mock_generate.return_value = "Score: 60/100"
        
        responses.add(
            responses.POST,
            "https://api.resend.com/emails",
            json={"error": "Invalid API key"},
            status=401
        )
        
        chats = ["Hello", "Hi"]
        result = send_email("test@example.com", chats)
        
        assert result is False
    
    @patch('simple_email.generate_analysis')
    def test_message_format_includes_research(self, mock_generate):
        """Test that the email message includes the research section"""
        mock_generate.return_value = "Analysis content here"
        
        with patch('simple_email.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            send_email("test@example.com", ["test", "message"])
            
            # Get the call arguments
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            
            html_content = request_data['html']
            
            # Check that research section is included
            assert "Research References" in html_content
            assert "Adolescent perspectives on depression as a disease of loneliness" in html_content
            assert "Kids helping kids: The lived experience of adolescents" in html_content
            assert "https://doi.org/10.1186/s13034-022-00481-y" in html_content
            assert "https://doi.org/10.1111/jcap.12299" in html_content
    
    @patch('simple_email.generate_analysis')
    def test_email_headers_and_structure(self, mock_generate):
        """Test email headers and basic structure"""
        mock_generate.return_value = "Test analysis"
        
        with patch('simple_email.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            send_email("user@test.com", ["msg1", "msg2"])
            
            # Check the request was made with correct headers
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            
            assert "Bearer re_Xu9Xyoig_5VLKuRJNSh51gZA16dtFKLB2" in headers['Authorization']
            assert headers['Content-Type'] == "application/json"
            
            # Check email structure
            request_data = call_args[1]['json']
            assert request_data['from'] == "onboarding@resend.dev"
            assert request_data['to'] == ["user@test.com"]
            assert request_data['subject'] == "Peer Support Analysis Report"
            assert 'html' in request_data


class TestEmailMessageFormatting:
    """Test the message formatting specifically"""
    
    @patch('simple_email.generate_analysis')
    @patch('simple_email.requests.post')
    def test_research_section_formatting(self, mock_post, mock_generate):
        """Test that the research section is properly formatted"""
        mock_generate.return_value = "LLM Response Here"
        mock_post.return_value.status_code = 200
        
        send_email("test@example.com", ["test"])
        
        call_args = mock_post.call_args
        html_content = call_args[1]['json']['html']
        
        # Check that the formatting is preserved with <pre> tags
        assert "<pre style='white-space: pre-wrap; font-family: Arial, sans-serif;'>" in html_content
        assert "---" in html_content  # Separator line
        assert "## Research References" in html_content
        assert "This analysis is based on the following peer-reviewed research:" in html_content
    
    @patch('simple_email.generate_analysis')
    @patch('simple_email.requests.post')
    def test_llm_response_before_research(self, mock_post, mock_generate):
        """Test that LLM response appears before research section"""
        mock_generate.return_value = "Score: 80/100\nDetailed analysis here"
        mock_post.return_value.status_code = 200
        
        send_email("test@example.com", ["test"])
        
        call_args = mock_post.call_args
        html_content = call_args[1]['json']['html']
        
        # Find positions in the content
        llm_pos = html_content.find("Score: 80/100")
        research_pos = html_content.find("Research References")
        
        assert llm_pos < research_pos, "LLM response should come before research section"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch('simple_email.generate_analysis')
    def test_empty_llm_response(self, mock_generate):
        """Test handling of empty LLM response"""
        mock_generate.return_value = ""
        
        with patch('simple_email.requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            result = send_email("test@example.com", ["test"])
            
            # Should still include research section even if LLM response is empty
            call_args = mock_post.call_args
            html_content = call_args[1]['json']['html']
            assert "Research References" in html_content
            assert result is True
    
    @patch('simple_email.generate_analysis')
    def test_network_error(self, mock_generate):
        """Test handling of network errors"""
        mock_generate.return_value = "Test response"
        
        with patch('simple_email.requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            # Should handle the exception gracefully
            with pytest.raises(Exception):
                send_email("test@example.com", ["test"])
    
    @patch('simple_email.generate_analysis')
    def test_invalid_email_address(self, mock_generate):
        """Test with invalid email address"""
        mock_generate.return_value = "Test response"
        
        with patch('simple_email.requests.post') as mock_post:
            mock_post.return_value.status_code = 400
            mock_post.return_value.text = "Invalid email address"
            
            result = send_email("invalid-email", ["test"])
            assert result is False


if __name__ == "__main__":
    # Run tests with: python -m pytest server/test_email.py -v
    pytest.main([__file__, "-v"])