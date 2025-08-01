# Turbo Alpaca 🦙 - AI-Powered Business Intelligence Platform

**HackED 2025 Submission**

An intelligent business analytics and idea generation platform that leverages AI to help entrepreneurs and business owners make data-driven decisions.

## Team: Depressed Divas 💪

- **Anshul Verma**
- **Siddharth Dileep**  
- **Pranav Gupta**
- **Ishaan Meena**
- **Melrita Cyriac**

---

## 🚀 Project Overview

Turbo Alpaca is a comprehensive web application built in 48 hours during HackED 2025. It combines multiple AI-powered features to assist businesses with:

- **Resume-Based Business Idea Generation**: Upload your resume and get personalized business ideas
- **Business Idea Evaluation**: Analyze your business concepts with AI-driven metrics
- **Intelligent Data Analytics**: Upload business data and get automated insights with visualizations
- **Competition Analysis**: Research competitors in your area using web scraping
- **Interactive AI Chatbot**: Get business advice and generate custom reports

## 🎯 Key Features

### 1. **AI Business Idea Generator**
- Upload your resume (PDF format)
- AI analyzes your skills and experience
- Generates personalized business ideas with step-by-step setup guides
- Powered by Groq's LLaMA models

### 2. **Business Idea Evaluation**
- Input your business concept and industry
- Get scored metrics on a radar chart:
  - Risk Assessment
  - Competitiveness
  - Setup Cost
  - Expected ROI
  - Scalability
- Detailed explanations for each metric

### 3. **Smart Data Analytics**
- Upload business data (CSV, JSON, PDF)
- Automatic data analysis and visualization generation
- Interactive dashboard with multiple chart types
- Insights on sales, customer patterns, inventory, and more

### 4. **Competition Analysis**
- Natural language queries like "analyze bakeries in Edmonton"
- Automated web scraping using Playwright
- Competitor data extraction and analysis
- Location-based business intelligence

### 5. **AI-Powered Chatbot**
- Context-aware business assistant
- Can generate plots and reports on demand
- Integrated with your uploaded data
- Supports multiple conversation threads

## 🛠️ Technology Stack

### Backend
- **Django 5.1.6** - Web framework
- **Python 3.11** - Core language
- **SQLite** - Database

### AI & ML
- **LangChain** - LLM orchestration framework
- **Groq API** - Fast LLM inference (LLaMA 3.3 70B, DeepSeek R1)
- **HuggingFace Embeddings** - Text embeddings
- **Pinecone** - Vector database for document storage

### Data Analysis
- **Pandas & NumPy** - Data manipulation
- **Plotly** - Interactive visualizations
- **Prophet** - Time series forecasting
- **scikit-learn** - Machine learning algorithms

### Web Scraping & Automation
- **Playwright** - Browser automation for competition analysis
- **BeautifulSoup** - HTML parsing
- **PyPDF2** - PDF text extraction

### Vector Storage & Retrieval
- **ChromaDB** - Alternative vector storage
- **FAISS** - Similarity search

### Frontend
- **TailwindCSS** - Styling framework
- **JavaScript** - Interactive features
- **HTML/CSS** - UI components

## 📁 Project Structure

```
turbo-alpaca/
├── turbo/                          # Main Django project
│   ├── turbo/                      # Project settings
│   │   ├── settings.py            # Django configuration
│   │   ├── urls.py                # URL routing
│   │   └── wsgi.py                # WSGI configuration
│   ├── upload/                     # Main app for business features
│   │   ├── views.py               # View controllers
│   │   ├── models.py              # Database models
│   │   ├── business.py            # Business idea generation
│   │   ├── business_idea_analysis.py  # Idea evaluation
│   │   ├── chatty.py              # AI chatbot logic
│   │   ├── competition_analysis.py    # Competitor research
│   │   ├── data_analysis_func.py      # Analytics functions
│   │   └── perform_analysis.py        # Data processing
│   ├── chatbot/                   # Chatbot app
│   ├── templates/                 # HTML templates
│   │   └── upload/               # Template files
│   ├── static/                   # Static assets
│   └── media/                    # User uploads & generated plots
├── data_for_chroma/              # Sample datasets
├── sample_data/                  # Test data files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anshcool5/turbo-alpaca.git
   cd turbo-alpaca
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-django-secret-key
   GROQ_API_KEY=your-groq-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   EMAIL_HOST_USER=your-email@gmail.com
   EMAIL_HOST_PASSWORD=your-email-password
   ```

5. **Run database migrations**
   ```bash
   cd turbo
   python manage.py migrate
   ```

6. **Create a superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:8000`

## 🔧 API Keys Setup

### Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account and generate an API key
3. Add it to your `.env` file

### Pinecone API Key
1. Visit [Pinecone](https://www.pinecone.io/)
2. Create an account and generate an API key
3. Add it to your `.env` file

## 📊 Usage Examples

### 1. Generate Business Ideas
1. Register/Login to the platform
2. Navigate to "Generate Idea"
3. Upload your resume (PDF format)
4. AI will analyze your skills and suggest relevant business ideas

### 2. Evaluate Business Concepts
1. Go to "Dashboard" → "Evaluate Idea"
2. Enter your business name, description, and industry
3. Get comprehensive metrics and insights

### 3. Analyze Business Data
1. Upload your business data (CSV/JSON/PDF)
2. Use the AI chatbot to request specific analyses:
   - "Show me revenue trends"
   - "Generate a customer segmentation analysis"
   - "Create a sales forecast"

### 4. Research Competition
1. Ask the chatbot: "Analyze coffee shops in downtown Toronto"
2. The system will automatically scrape relevant data
3. Get insights about competitor density and market opportunities

## 🎨 Features Deep Dive

### Business Idea Generation
- **Input**: Resume PDF
- **Processing**: LLM analyzes skills, experience, and background
- **Output**: Structured business ideas with implementation steps
- **Example**: Software developer → SaaS product recommendations

### Idea Evaluation Metrics
- **Risk Score**: Market volatility, regulatory challenges
- **Competitiveness**: Market saturation, differentiation potential
- **Setup Cost**: Initial investment requirements
- **ROI Potential**: Expected return timeline and profitability
- **Scalability**: Growth potential and expansion opportunities

### Data Analytics Capabilities
- Revenue analysis and forecasting
- Customer segmentation using K-means clustering
- Inventory management and stock optimization
- Seasonal trend analysis
- Profit margin calculations
- Customer churn prediction

### Competition Analysis
- Location-based competitor research
- Automated web scraping of business directories
- Market density analysis
- Competitor feature comparison

## 🛡️ Security Features

- User authentication and authorization
- File upload validation and sanitization
- CSRF protection
- Secure API key management
- Database query protection

## 🔮 Future Enhancements

- [ ] **Multi-language Support**: Expand beyond English
- [ ] **Mobile App**: React Native companion app
- [ ] **Advanced ML Models**: Custom-trained business intelligence models
- [ ] **Real-time Collaboration**: Team workspace features
- [ ] **Integration APIs**: Connect with popular business tools
- [ ] **Advanced Visualization**: 3D charts and interactive dashboards
- [ ] **Market Research**: Real-time market data integration
- [ ] **Financial Modeling**: Advanced financial projection tools

## 🤝 Contributing

This was a 48-hour hackathon project, but we welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Hackathon Achievement

Built in **48 hours** during HackED 2025 at the University of Alberta. This project demonstrates:
- Rapid prototyping capabilities
- Integration of multiple AI technologies
- Full-stack web development
- Creative problem-solving under time constraints

## 📞 Contact & Support

- **GitHub**: [@Anshcool5](https://github.com/Anshcool5)
- **Project Demo**: [turbo-alpaca.duckdns.org](https://turbo-alpaca.duckdns.org)

---

**Made with ❤️ by Team Depressed Divas during HackED 2025**

*"Empowering entrepreneurs with AI-driven business intelligence"*
