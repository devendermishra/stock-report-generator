# LangChain Import Fix Guide

## 🚨 Issue: Import Errors

The original code had import errors because LangChain has changed its import structure in newer versions. Here's how to fix it:

## 🔧 Solution 1: Install Correct Dependencies

### **Quick Fix - Run Installation Script**
```bash
# Make script executable and run
chmod +x install_langchain.sh
./install_langchain.sh
```

### **Manual Installation**
```bash
# Install core LangChain packages
pip install langchain>=0.1.0
pip install langchain-core>=0.1.0
pip install langchain-openai>=0.1.0
pip install langchain-community>=0.1.0

# Install LangGraph
pip install langgraph>=0.1.0

# Install OpenAI integration
pip install openai>=1.0.0
```

## 🔧 Solution 2: Use Fixed Import Version

I've created a fixed version of the LangGraph agent that handles import compatibility:

### **File: `src/agents/langgraph_dynamic_agent_fixed.py`**

This version includes:
- **Compatible imports** for different LangChain versions
- **Fallback mechanisms** when imports fail
- **Error handling** for missing dependencies
- **Direct tool execution** as fallback

## 🔧 Solution 3: Update Import Statements

### **Original (Broken) Imports**
```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
```

### **Fixed Imports**
```python
# Try different import paths for compatibility
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        print("LangChain prompts not available. Please install: pip install langchain-core")
        raise

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
except ImportError:
    try:
        from langchain.agents import create_agent, AgentExecutor
        create_openai_functions_agent = create_agent
    except ImportError:
        print("LangChain agents not available. Please install: pip install langchain")
        raise
```

## 🎯 Usage Options

### **Option 1: Use Fixed Version**
```python
# Import the fixed version
from src.agents.langgraph_dynamic_agent_fixed import LangGraphDynamicAgentFixed

# Use in your code
agent = LangGraphDynamicAgentFixed(
    agent_id="fixed_agent",
    mcp_context=mcp_context,
    stock_data_tool=stock_data_tool,
    web_search_tool=web_search_tool,
    summarizer_tool=summarizer_tool,
    openai_api_key=openai_api_key,
    model="gpt-4o-mini"
)
```

### **Option 2: Update Main.py to Use Fixed Version**
```python
# In src/main.py, change the import
from .agents.langgraph_dynamic_agent_fixed import LangGraphDynamicAgentFixed as LangGraphDynamicAgent
```

### **Option 3: Use Traditional or Dynamic Mode**
If LangGraph continues to have issues, you can use the other modes:

```bash
# Traditional fixed-sequence (no LangChain required)
python src/main.py --symbol RELIANCE

# Dynamic custom framework (no LangChain required)
python src/main.py --symbol RELIANCE --dynamic
```

## 🔍 Troubleshooting

### **Error: "cannot find import of from langchain.agents"**
**Solution**: Install the correct LangChain version:
```bash
pip install langchain>=0.1.0 langchain-core>=0.1.0
```

### **Error: "cannot find import of from langchain.prompts"**
**Solution**: Use the core imports:
```bash
pip install langchain-core>=0.1.0
```

### **Error: "LangGraph not available"**
**Solution**: Install LangGraph:
```bash
pip install langgraph>=0.1.0
```

### **Error: "OpenAI not available"**
**Solution**: Install OpenAI integration:
```bash
pip install langchain-openai>=0.1.0 openai>=1.0.0
```

## 📋 Version Compatibility

| LangChain Version | Import Path | Status |
|-------------------|-------------|---------|
| 0.1.x+ | `langchain_core.prompts` | ✅ Recommended |
| 0.0.x | `langchain.prompts` | ⚠️ Legacy |
| 0.1.x+ | `langchain_core.messages` | ✅ Recommended |
| 0.0.x | `langchain.schema` | ⚠️ Legacy |

## 🚀 Quick Start (Fixed Version)

### **1. Install Dependencies**
```bash
./install_langchain.sh
```

### **2. Test Installation**
```python
# Test imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.agents import create_openai_functions_agent
    from langgraph import StateGraph
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### **3. Use LangGraph Agent**
```bash
# Use the fixed version
python src/main.py --symbol RELIANCE --langgraph --verbose
```

## 🎯 Alternative: Use Without LangGraph

If you continue to have issues with LangGraph, you can use the other dynamic agent systems:

### **Dynamic Custom Framework**
```bash
python src/main.py --symbol RELIANCE --dynamic
```

### **Traditional Fixed-Sequence**
```bash
python src/main.py --symbol RELIANCE
```

Both of these work without LangChain dependencies and still provide intelligent, LLM-driven analysis!

## 🎉 Summary

The import issues are fixed by:

1. **Installing correct dependencies** with the installation script
2. **Using the fixed version** with compatible imports
3. **Fallback mechanisms** when imports fail
4. **Alternative modes** that don't require LangChain

The system now works with multiple LangChain versions and provides graceful fallbacks! 🚀

