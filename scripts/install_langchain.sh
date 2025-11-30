#!/bin/bash

# Installation script for LangChain and LangGraph dependencies
# This script installs the correct versions for the dynamic agent system

echo "🚀 Installing LangChain and LangGraph dependencies..."

# Install core LangChain packages
echo "📦 Installing LangChain core packages..."
pip install langchain>=0.1.0
pip install langchain-core>=0.1.0
pip install langchain-openai>=0.1.0
pip install langchain-community>=0.1.0

# Install LangGraph
echo "📦 Installing LangGraph..."
pip install langgraph>=0.1.0

# Install OpenAI integration
echo "📦 Installing OpenAI integration..."
pip install openai>=1.0.0

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install pydantic>=2.0.0
pip install typing-extensions>=4.0.0

echo "✅ LangChain and LangGraph dependencies installed successfully!"
echo ""
echo "🔧 To use the LangGraph dynamic agent system:"
echo "   python src/main.py --symbol RELIANCE --langgraph"
echo ""
echo "📋 Available report types:"
echo "   --symbol RELIANCE                    # Traditional fixed-sequence"
echo "   --symbol RELIANCE --dynamic          # Custom dynamic framework"
echo "   --symbol RELIANCE --langgraph        # LangGraph framework"

