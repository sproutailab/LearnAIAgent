# 8-Week AI Agent Learning Plan

## Week 1: Foundations - Developer Skills & Linear Algebra

### The Missing Semester of Your CS Education
- **Course Link**: https://missing.csail.mit.edu/
- **Key Topics**:
  - Shell Tools and Scripting
  - Editors (Vim)
  - Data Wrangling
  - Version Control (Git)
  - Debugging and Profiling
- **Additional Resources**:
  - GitHub Learning Lab: https://lab.github.com/
  - Oh My Zsh for shell productivity: https://ohmyz.sh/

### Linear Algebra for AI
- **Primary Resource**: MIT OpenCourseWare 18.06 Linear Algebra with Prof. Gilbert Strang
  - Course link: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
  - Focus lectures: 1-8 (Foundations through Eigenvalues)
- **Supplementary Resources**: 
  - 3Blue1Brown's "Essence of Linear Algebra" series: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
  - Python NumPy tutorial for implementing matrix operations: https://numpy.org/doc/stable/user/absolute_beginners.html
- **Hands-on Practice**: 
  - Implement matrix operations from scratch in Python
  - Complete MIT OCW problem sets 1-2

## Week 2: Machine Learning & Deep Learning Basics

### Machine Learning Basics
- **Primary Course**: Google's "Machine Learning Crash Course"
  - Course link: https://developers.google.com/machine-learning/crash-course
  - Focus modules: 
    - ML Concepts
    - Reducing Loss
    - Classification
    - Regularization
    - Feature Engineering
- **Additional Resources**:
  - Scikit-learn tutorials: https://scikit-learn.org/stable/tutorial/index.html
  - "Introduction to Statistical Learning" (free book): https://www.statlearning.com/
- **Hands-on Project**:
  - Implement fashion item classification model using Fashion MNIST dataset
  - Resource: https://github.com/zalandoresearch/fashion-mnist

### Deep Learning Introduction & Web Scraping
- **Primary Course**: Fast.ai "Practical Deep Learning for Coders" (Part 1)
  - Course link: https://course.fast.ai/
  - Focus lessons: 1-3
- **Web Scraping Resources**:
  - Beautiful Soup documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
  - Selenium Python documentation: https://selenium-python.readthedocs.io/
  - "Web Scraping with Python" by Ryan Mitchell (reference)
- **Project: Fashion Data Collection**
  - Build a web scraper for fashion websites (Zara, H&M, ASOS)
  - Implement proper labeling system (categories, styles, colors)
  - Create dataset of 1,000+ labeled fashion items
  - Resources:
    - Ethical web scraping guide: https://www.scrapehero.com/how-to-prevent-getting-blacklisted-while-scraping/
    - Clothing attribute dataset: https://github.com/openai/CLIP/blob/main/data/prompts.md

## Week 3: AI Agent Fundamentals

### DeepLearning.AI "Building AI Agents with LLMs" (2024)
- **Course Link**: https://www.deeplearning.ai/courses/building-ai-agents-with-llms/
- **Key Modules**:
  - Agent architecture and design patterns
  - Tool use and function calling
  - Planning and reasoning
  - Evaluation and debugging
- **Additional Resources**:
  - LangChain documentation: https://python.langchain.com/docs/get_started/introduction
  - OpenAI Cookbook (function calling): https://cookbook.openai.com/
  - Paper: "ReAct: Synergizing Reasoning and Acting in LLMs"
- **Hands-on Projects**:
  - Implement a simple search agent with tool use
  - Build a web research assistant

### Microsoft AutoGen Course (2024)
- **Course Link**: https://learn.microsoft.com/en-us/training/modules/introduction-to-autogen/
- **Key Modules**:
  - Multi-agent architectures
  - Agent communication protocols
  - Human-in-the-loop systems
- **Additional Resources**:
  - AutoGen GitHub: https://github.com/microsoft/autogen
  - Paper: "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"
- **Hands-on Project**:
  - Create a simple multi-agent system with specialized roles

## Week 4: RAG & Advanced Agent Techniques

### RAG (Retrieval-Augmented Generation) Fundamentals
- **Primary Course**: DeepLearning.AI "Building RAG Applications" (2024)
  - Course link: https://www.deeplearning.ai/short-courses/building-rag-applications/
- **Key Modules**:
  - Document ingestion and chunking
  - Embedding models and vector databases
  - Retrieval strategies
  - Reranking and filtering
- **Additional Resources**:
  - LlamaIndex documentation: https://docs.llamaindex.ai/en/stable/
  - Pinecone "Advanced RAG" course: https://www.pinecone.io/learn/course/advanced-rag/
  - Langchain RAG cookbook: https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_cookbook.ipynb
- **Hands-on Project**:
  - Build a RAG system for fashion knowledge using your scraped dataset

### HuggingFace Agent Building with Transformers
- **Course Materials**: https://huggingface.co/learn/nlp-course/chapter0/1
- **Key Modules**:
  - Transformer architecture fundamentals
  - Fine-tuning models
  - Building agents with HF transformers
- **Additional Resources**:
  - Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/index
  - Agent examples: https://huggingface.co/docs/transformers/transformers_agents
- **Hands-on Project**:
  - Implement an agent with HuggingFace Transformers

## Week 5: LLM Integration & Project Planning

### Large Language Models & API Integration
- **Key Resources**:
  - OpenAI API documentation: https://platform.openai.com/docs/introduction
  - Anthropic Claude API documentation: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
- **Key Topics**:
  - Model selection and parameters
  - Function calling and tool use
  - System prompts and few-shot learning
  - Cost optimization
- **Hands-on Project**:
  - Build a simple fashion trend analyzer with OpenAI or Claude

### Fashion Style Assistant Project Planning
- **Activities**:
  - Requirements gathering and system design
  - Architecture planning
  - Tools and frameworks selection
- **Resources**:
  - System design primer: https://github.com/donnemartin/system-design-primer
  - UI/UX design resources: https://www.figma.com/community/
  - API documentation for fashion retailers
- **Deliverable**:
  - Detailed project plan with architecture diagram
  - GitHub repository setup with initial code structure

## Week 6: Fashion Style Assistant - Core Implementation

### User Profile System
- **Implementation Components**:
  - Profile creation with gender, age, body type inputs
  - Style preferences collection
  - Size preferences for different clothing categories
  - Budget range and brand preferences
- **Resources**:
  - User modeling tutorials
  - Firebase/Supabase for user management
  - Fashion size standardization guides

### Web Scraping & Database Integration
- **Implementation Components**:
  - Enhance your web scraper from Week 2
  - Implement scheduled data collection from fashion websites
  - Create vector embeddings for fashion items
  - Build database for fashion items and user profiles
- **Resources**:
  - Vector databases: Pinecone, Chroma, or Weaviate
  - CLIP for visual embeddings: https://github.com/openai/CLIP
  - E-commerce API integrations (if available)

## Week 7: Fashion Style Assistant - Agent & UI Development

### AI Agent Implementation
- **Implementation Components**:
  - Style analysis functions
  - Celebrity style reference system
  - Outfit recommendation engine
  - RAG system for fashion knowledge
- **Resources**:
  - LangChain Agents documentation
  - OpenAI function calling examples
  - AutoGen multi-agent frameworks

### Image Generation & UI Development
- **Implementation Components**:
  - Outfit preview generation
  - Style transfer techniques
  - User interface development
- **Resources**:
  - Streamlit documentation: https://docs.streamlit.io/
  - Stable Diffusion guides: https://stability.ai/blog/stable-diffusion-public-release
  - Hugging Face Diffusers library: https://huggingface.co/docs/diffusers/index

## Week 8: Project Refinement & Completion

### Advanced Features Implementation
- **Implementation Components**:
  - Budget filtering
  - Brand preference system
  - Occasion-specific recommendation module
  - Weather API integration for contextual suggestions
- **Resources**:
  - Weather APIs: OpenWeatherMap, WeatherAPI
  - Price comparison techniques
  - Brand categorization data

### Testing, Documentation & Presentation
- **Activities**:
  - User testing with sample profiles
  - Performance optimization
  - Documentation creation
  - Project presentation preparation
- **Resources**:
  - Documentation tools: MkDocs, Sphinx
  - Deployment options: Heroku, AWS, Google Cloud
  - Presentation frameworks: Reveal.js, Slides.com

## AI Tools

### AI Models
1. **Large Language Models (LLMs)**
   - **OpenAI GPT Models**
     - GPT-4o: https://platform.openai.com/docs/models/gpt-4o
     - GPT-4: https://platform.openai.com/docs/models/gpt-4
     - Usage documentation: https://platform.openai.com/docs/guides/text-generation
   - **Anthropic Claude Models**
     - Claude 3 Opus/Sonnet/Haiku: https://docs.anthropic.com/claude/docs/models-overview
     - Claude API documentation: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
   - **Open Source Models**
     - Llama 3: https://llama.meta.com/
     - Mistral AI: https://mistral.ai/
     - Together AI models: https://www.together.ai/
   - **Multimodal Models**
     - GPT-4o Vision: https://platform.openai.com/docs/guides/vision
     - Claude 3 Sonnet Vision: https://docs.anthropic.com/claude/docs/vision
     - DALL-E 3: https://openai.com/dall-e-3
     - Midjourney: https://www.midjourney.com/
     - Stable Diffusion: https://stability.ai/stable-diffusion

2. **Embedding Models**
   - OpenAI embeddings: https://platform.openai.com/docs/guides/embeddings
   - HuggingFace Sentence Transformers: https://www.sbert.net/
   - CLIP (for images and text): https://github.com/openai/CLIP
   - Cohere Embed: https://cohere.com/embeddings

3. **Voice & Speech Models**
   - OpenAI Whisper (speech-to-text): https://github.com/openai/whisper
   - ElevenLabs (text-to-speech): https://elevenlabs.io/
   - VALL-E X: https://valle-demo.github.io/

### AI Coding Tools
1. **Code Assistants**
   - GitHub Copilot: https://github.com/features/copilot
   - Anthropic Claude Code Interpreter: https://www.anthropic.com/news/claude-code-interpreter
   - Replit Ghostwriter: https://replit.com/ghostwriter
   - Amazon CodeWhisperer: https://aws.amazon.com/codewhisperer/
   - Codeium: https://codeium.com/

2. **Development Environments**
   - JupyterLab + AI extensions: https://jupyterlab.readthedocs.io/
   - VS Code + GitHub Copilot: https://marketplace.visualstudio.com/items?itemName=GitHub.copilot
   - CodeGeeX VS Code plugin: https://marketplace.visualstudio.com/items?itemName=aminer.codegeex

3. **Specialized AI Development Tools**
   - Hugging Face Spaces: https://huggingface.co/spaces
   - Streamlit: https://streamlit.io/
   - Gradio: https://gradio.app/
   - LangChain: https://www.langchain.com/
   - LlamaIndex: https://www.llamaindex.ai/

4. **AI Testing & Debugging Tools**
   - DeepChecks: https://deepchecks.com/
   - WhyLabs: https://whylabs.ai/
   - Weights & Biases: https://wandb.ai/
   - Arize AI: https://arize.com/

5. **AI Deployment Platforms**
   - Hugging Face: https://huggingface.co/
   - Replicate: https://replicate.com/
   - Modal: https://modal.com/
   - SageMaker: https://aws.amazon.com/sagemaker/
   - Runhouse: https://run.house/

## Additional Resources & References

### Machine Learning & Deep Learning
1. CS231n: Convolutional Neural Networks for Visual Recognition: http://cs231n.stanford.edu/
2. Deep Learning with PyTorch: https://pytorch.org/tutorials/
3. TensorFlow tutorials: https://www.tensorflow.org/tutorials
4. Papers With Code (Fashion category): https://paperswithcode.com/task/fashion-image-retrieval

### AI Agents
1. BabyAGI repository: https://github.com/yoheinakajima/babyagi
2. LangGraph documentation: https://langchain-ai.github.io/langgraph/
3. Gorilla: Large Language Model Connected with Massive APIs: https://gorilla.cs.berkeley.edu/
4. AgentForge framework: https://github.com/DataBassGit/AgentForge

### RAG Systems
1. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" paper: https://arxiv.org/abs/2005.11401
2. Weaviate vector database: https://weaviate.io/developers/weaviate
3. Haystack framework: https://haystack.deepset.ai/
4. Semantic search guide: https://www.sbert.net/examples/applications/semantic-search/README.html

### Fashion & Style Technology
1. Fashion-Gen dataset: https://github.com/facebookresearch/FashionGen
2. DeepFashion database: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
3. Style-based GAN for outfit generation: https://github.com/NVlabs/stylegan3
4. "FashionBERT: Text and Image Matching with Adaptive Loss for Cross-modal Retrieval" paper

### Development Tools
1. Visual Studio Code with AI extensions: https://marketplace.visualstudio.com/vscode
2. Jupyter Lab for interactive development: https://jupyter.org/
3. Postman for API testing: https://www.postman.com/
4. Docker for containerization: https://www.docker.com/
