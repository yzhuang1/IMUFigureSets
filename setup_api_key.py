"""
Setup script to configure OpenAI API key
Run this script to securely set up your API key
"""

import os
import sys

def setup_api_key():
    """Interactive setup for OpenAI API key"""
    print("OpenAI API Key Setup")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Get API key from user
    print("\nPlease enter your OpenAI API key:")
    print("(You can find it at: https://platform.openai.com/api-keys)")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't start with 'sk-'. Please verify it's correct.")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Setup cancelled.")
            return
    
    # Create .env file
    try:
        with open('.env', 'w') as f:
            f.write(f"# OpenAI Configuration\n")
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write(f"\n")
            f.write(f"# Optional: Customize OpenAI settings\n")
            f.write(f"OPENAI_BASE_URL=https://api.openai.com/v1\n")
            f.write(f"OPENAI_MODEL=gpt-5\n")
        
        print("✅ .env file created successfully!")
        print("🔒 Your API key is now securely stored in .env (this file is gitignored)")
        print("\nYou can now use the AI-enhanced features of the ML pipeline.")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return

def test_configuration():
    """Test if the configuration is working"""
    print("\nTesting configuration...")
    
    try:
        from config import config
        
        if config.is_openai_configured():
            print("✅ OpenAI configuration is valid")
            print(f"   Model: {config.openai_model}")
            print(f"   Base URL: {config.openai_base_url}")
            return True
        else:
            print("❌ OpenAI configuration is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

if __name__ == "__main__":
    setup_api_key()
    
    if test_configuration():
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n❌ Setup failed. Please check your configuration.")
        sys.exit(1)
