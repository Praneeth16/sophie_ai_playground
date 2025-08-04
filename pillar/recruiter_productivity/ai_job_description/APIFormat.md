# AI Job Description Generator API

## Overview
This API provides AI-powered job description generation using multiple specialized agents. It takes basic job requirements and company information as input and returns a professionally optimized job description in markdown format.

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and healthy.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "1641234567.890"
}
```

---

### 2. Generate Job Description
**POST** `/generate-job-description`

Generate a complete job description using AI agents.

#### Request Body
```json
{
  "job_title": "Senior Backend Engineer",
  "company_name": "TechCorp Inc.",
  "recruiter_summary": "We need a senior backend engineer with 5+ years experience in Python and distributed systems. The role involves architecting scalable microservices, mentoring junior developers, and collaborating with product teams. Must have experience with AWS, Docker, and PostgreSQL. Remote-friendly with quarterly team meetups.",
  "salary_info": "$120,000 - $150,000 + equity + benefits",
  "remote_policy": "Remote-first with quarterly team meetups",
  "location": "San Francisco, CA",
  "model": "moonshotai/kimi-k2",
  "temperature": 0.7
}
```

#### Request Schema
| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `job_title` | string | ✅ | The job title for the position | "Senior Backend Engineer" |
| `company_name` | string | ✅ | Company name for research and personalization | "TechCorp Inc." |
| `recruiter_summary` | string | ✅ | Hiring manager brief with role requirements and context | "We need a senior backend engineer..." |
| `salary_info` | string | ❌ | Salary range or compensation details | "$120,000 - $150,000 + equity" |
| `remote_policy` | string | ❌ | Work arrangement policy | "Remote-first", "Hybrid", "On-site" |
| `location` | string | ❌ | Job location | "San Francisco, CA" |
| `model` | string | ❌ | AI model to use (default: "moonshotai/kimi-k2") | "moonshotai/kimi-k2" |
| `temperature` | float | ❌ | AI generation temperature 0.0-2.0 (default: 0.7) | 0.7 |

#### Success Response (200)
```json
{
  "success": true,
  "job_description": "## Senior Backend Engineer\n\n### About TechCorp Inc.\nTechCorp Inc. is a leading technology company...\n\n### The Role\nWe are seeking an experienced Senior Backend Engineer...\n\n### What You'll Do\n- Architect and develop scalable microservices\n- Mentor junior developers and provide technical leadership\n- Collaborate with product teams to deliver features\n\n### What You'll Bring\n- 5+ years of backend development experience\n- Strong proficiency in Python and distributed systems\n- Experience with AWS, Docker, and PostgreSQL\n\n### Compensation & Benefits\n$120,000 - $150,000 + equity + comprehensive benefits\n\n### Work Arrangement\nRemote-first culture with quarterly team meetups\n\n### Location\nSan Francisco, CA (Remote welcome)\n\n### Why Join Us?\nJoin our innovative team and make a direct impact...",
  "error": null,
  "metadata": {
    "processing_time_seconds": 12.34,
    "model_used": "moonshotai/kimi-k2",
    "company_research_available": true,
    "timestamp": "1641234567.890"
  }
}
```

#### Error Response (400/500)
```json
{
  "success": false,
  "job_description": null,
  "error": "An error occurred during generation: API Request Error",
  "metadata": null
}
```

#### Response Schema
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the generation was successful |
| `job_description` | string\|null | Generated job description in markdown format |
| `error` | string\|null | Error message if generation failed |
| `metadata` | object\|null | Additional information about the generation process |
| `metadata.processing_time_seconds` | float | Time taken to process the request |
| `metadata.model_used` | string | AI model that was used |
| `metadata.company_research_available` | boolean | Whether company research data was found |
| `metadata.timestamp` | string | Generation timestamp |

---

### 3. Get Available Models
**GET** `/models`

Get list of available AI models for job description generation.

#### Response
```json
{
  "models": [
    "moonshotai/kimi-k2",
    "openai/gpt-4o", 
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.1-8b-instruct"
  ],
  "default": "moonshotai/kimi-k2"
}
```

## AI Agent Pipeline

The API uses a 4-stage AI agent pipeline:

1. **Experience Extraction Agent** - Analyzes role requirements and seniority level
2. **Market Research Agent** - Conducts company research and competitive analysis
3. **Job Description Architect** - Creates the initial structured job description
4. **Optimization Agent** - Optimizes for ATS compatibility, SEO, and candidate appeal

## Features

- **Company Intelligence**: Automatic company research using web search
- **Market Analysis**: Industry context and competitive insights
- **ATS Optimization**: Optimized for Applicant Tracking Systems
- **SEO Enhancement**: Better visibility on job boards
- **Markdown Output**: Clean, formatted output ready for publishing
- **Caching**: Intelligent caching for company research data
- **Multiple Models**: Support for various AI models

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 500 | Internal Server Error - API configuration or generation error |

## Environment Requirements

- `OPENROUTER_API_KEY`: Required for AI model access

## Usage Examples

### Python (requests)
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-job-description",
    json={
        "job_title": "Senior Backend Engineer",
        "company_name": "TechCorp Inc.",
        "recruiter_summary": "We need a senior backend engineer...",
        "salary_info": "$120,000 - $150,000",
        "remote_policy": "Remote-first",
        "location": "San Francisco, CA"
    }
)

data = response.json()
if data["success"]:
    print(data["job_description"])
else:
    print(f"Error: {data['error']}")
```

### JavaScript (fetch)
```javascript
const response = await fetch('http://localhost:8000/generate-job-description', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        job_title: "Senior Backend Engineer",
        company_name: "TechCorp Inc.",
        recruiter_summary: "We need a senior backend engineer...",
        salary_info: "$120,000 - $150,000",
        remote_policy: "Remote-first",
        location: "San Francisco, CA"
    })
});

const data = await response.json();
if (data.success) {
    console.log(data.job_description);
} else {
    console.error('Error:', data.error);
}
```

### cURL
```bash
curl -X POST "http://localhost:8000/generate-job-description" \
     -H "Content-Type: application/json" \
     -d '{
       "job_title": "Senior Backend Engineer",
       "company_name": "TechCorp Inc.", 
       "recruiter_summary": "We need a senior backend engineer...",
       "salary_info": "$120,000 - $150,000",
       "remote_policy": "Remote-first",
       "location": "San Francisco, CA"
     }'
```

## Performance Notes

- Average response time: 10-15 seconds
- Company research is cached for efficiency
- Use lower temperature values (0.1-0.3) for more consistent results
- Use higher temperature values (0.7-1.0) for more creative descriptions

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing rate limiting for production use.