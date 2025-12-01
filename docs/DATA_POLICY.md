# Data Policy Documentation

## User-Related Data Handling Policy

**Last Updated:** [Current Date]  
**Version:** 1.0  
**Effective Date:** [Current Date]

---

## Table of Contents

1. [Introduction](#introduction)
2. [Scope and Applicability](#scope-and-applicability)
3. [Data Collection](#data-collection)
4. [Data Usage](#data-usage)
5. [Data Storage and Retention](#data-storage-and-retention)
6. [Data Security](#data-security)
7. [Data Sharing and Third-Party Services](#data-sharing-and-third-party-services)
8. [User Rights](#user-rights)
9. [Data Deletion and Retention](#data-deletion-and-retention)
10. [Compliance](#compliance)
11. [Contact Information](#contact-information)
12. [Policy Updates](#policy-updates)

---

## Introduction

This Data Policy Documentation outlines how the Stock Report Generator system collects, uses, stores, and protects user-related data. This policy is designed to ensure transparency, security, and compliance with applicable data protection regulations.

### Purpose

The Stock Report Generator is an educational multi-agent AI system that generates equity research reports for NSE stocks. This policy document provides clear information about:

- What data is collected
- How data is used
- Where data is stored
- How data is protected
- User rights regarding their data

### Important Notice

**This project is for educational purposes only. Do not use its output as stock recommendation.**

**This project uses yfinance python project which is intended for personal purpose only. Do not use this project in production or commercial applications.**

---

## Scope and Applicability

### What This Policy Covers

This policy applies to all user-related data handled by the Stock Report Generator system, including:

- User inputs (stock symbols, company names, sectors)
- Generated reports and analysis outputs
- System logs and execution traces
- Configuration data (API keys)
- Temporary and processed data files

### What This Policy Does Not Cover

This policy does not cover:

- Publicly available stock market data retrieved from external sources (Yahoo Finance, NSE)
- Third-party service provider data policies (OpenAI, Yahoo Finance, DuckDuckGo)
- Data stored outside the system's control (user's local filesystem, cloud storage)

### System Components Covered

- **Agents**: All specialized agents and AI-powered iterative agents
- **Tools**: All 15integrated tools
- **Storage**: Local filesystem storage (reports, logs, data directories)
- **External Services**: API integrations (OpenAI, Yahoo Finance, DuckDuckGo)

---

## Data Collection

### Types of Data Collected

#### 1. User Input Data

**Stock Symbol**
- **Type**: Text string (e.g., "RELIANCE", "TCS", "HDFCBANK")
- **Purpose**: Primary input to generate stock research reports
- **Collection Method**: Command-line argument or programmatic API call
- **Required**: Yes
- **Sensitive**: No (public stock symbols)

**Company Name**
- **Type**: Text string (e.g., "Reliance Industries Limited")
- **Purpose**: Optional input to enhance report accuracy
- **Collection Method**: Command-line argument or programmatic API call
- **Required**: No (auto-detected if not provided)
- **Sensitive**: No (public company information)

**Sector Information**
- **Type**: Text string (e.g., "Oil & Gas", "Technology", "Banking")
- **Purpose**: Optional input for sector-specific analysis
- **Collection Method**: Command-line argument or programmatic API call
- **Required**: No (auto-detected if not provided)
- **Sensitive**: No (public sector classification)

#### 2. Configuration Data

**API Keys**
- **Type**: Secret string (OpenAI API key)
- **Purpose**: Authentication for external AI services
- **Collection Method**: Environment variable or `.env` file
- **Required**: Yes (for AI-powered features)
- **Sensitive**: Yes (highly sensitive)
- **Storage**: Local `.env` file (not committed to version control)
- **Access**: Only used for API authentication, never logged or exposed

#### 3. Generated Data

**Research Reports**
- **Type**: Markdown and PDF files
- **Purpose**: Output of stock analysis
- **Collection Method**: System-generated
- **Storage Location**: `reports/` directory
- **Sensitive**: No (contains only public stock analysis)
- **Retention**: Stored locally until manually deleted

**Analysis Data**
- **Type**: JSON, structured data
- **Purpose**: Intermediate analysis results
- **Collection Method**: System-generated during workflow execution
- **Storage Location**: `data/processed/`, `data/outputs/`
- **Sensitive**: No (contains only public stock data)
- **Retention**: Stored locally until manually deleted

#### 4. System Logs

**Execution Logs**
- **Type**: Text log files
- **Purpose**: System debugging, error tracking, performance monitoring
- **Collection Method**: Automatic logging during system execution
- **Storage Location**: `logs/stock_report_generator.log`
- **Sensitive**: Contains API call details, prompts, and responses
- **Retention**: Stored locally, rotated based on size

**Prompt Logs**
- **Type**: Text log files
- **Purpose**: Tracking LLM prompts and responses for debugging
- **Collection Method**: Automatic logging of LLM interactions
- **Storage Location**: `logs/prompts.log`
- **Sensitive**: Contains detailed prompts and AI responses
- **Retention**: Stored locally, rotated based on size

#### 5. Temporary Data

**Temporary Files**
- **Type**: Various file formats
- **Purpose**: Intermediate processing files
- **Collection Method**: System-generated during execution
- **Storage Location**: `temp/` directory
- **Sensitive**: May contain processed stock data
- **Retention**: Cleared after report generation or manually deleted

**Session Data**
- **Type**: In-memory state data
- **Purpose**: Workflow state management during execution
- **Collection Method**: Runtime state tracking
- **Storage**: In-memory only (not persisted)
- **Sensitive**: Contains workflow execution context
- **Retention**: Cleared after workflow completion

### Data Collection Methods

1. **Direct User Input**: Command-line arguments, programmatic API calls
2. **Environment Variables**: Configuration and API keys from `.env` file
3. **Automatic Generation**: System-generated reports, logs, and analysis data
4. **External Data Retrieval**: Stock data from Yahoo Finance, NSE (public data)

### Data Minimization

The system follows data minimization principles:

- Only collects data necessary for report generation
- Does not collect personal identifying information (PII)
- Does not track user behavior or usage patterns
- Does not collect location data or device information
- Does not use cookies or tracking technologies

---

## Data Usage

### Primary Use Cases

#### 1. Report Generation

**Purpose**: Generate comprehensive equity research reports

**Data Used**:
- Stock symbol (required)
- Company name (optional, auto-detected)
- Sector information (optional, auto-detected)
- External stock data (Yahoo Finance, NSE)
- AI-generated analysis (OpenAI API)

**Processing**:
- Multi-agent workflow execution
- Data retrieval and analysis
- Report synthesis and formatting
- PDF generation

#### 2. System Operation

**Purpose**: System functionality, debugging, and monitoring

**Data Used**:
- Configuration data (API keys for authentication)
- Execution logs for error tracking
- Performance metrics for optimization

**Processing**:
- API authentication
- Error logging and debugging
- Performance monitoring

### Data Processing Activities

1. **Stock Data Retrieval**: Fetching real-time and historical stock data
2. **Web Search**: Searching for company news, sector trends, market intelligence
3. **Financial Analysis**: Calculating ratios, metrics, and financial indicators
4. **AI Analysis**: Using LLM for comprehensive analysis and report generation
5. **Report Formatting**: Converting analysis into professional markdown and PDF formats

### Legal Basis for Processing

- **Legitimate Interest**: System operation and report generation
- **User Consent**: Implied through system usage
- **Contractual Necessity**: Required for system functionality

### Data Usage Limitations

- **No Commercial Use**: Data is not used for commercial purposes
- **No Third-Party Marketing**: Data is not shared for marketing purposes
- **No User Profiling**: No user behavior tracking or profiling
- **No Data Selling**: User data is never sold to third parties

---

## Data Storage and Retention

### Storage Locations

#### 1. Local Filesystem Storage

**Reports Directory** (`reports/`)
- **Content**: Generated PDF and Markdown reports
- **Format**: `.pdf`, `.md` files
- **Naming**: `{symbol}_{timestamp}.{ext}`
- **Access**: Local filesystem access only
- **Backup**: User's responsibility

**Logs Directory** (`logs/`)
- **Content**: System execution logs, prompt logs
- **Format**: `.log` files
- **Naming**: `stock_report_generator.log`, `prompts.log`
- **Access**: Local filesystem access only
- **Rotation**: Based on file size (configurable)

**Data Directory** (`data/`)
- **Subdirectories**:
  - `inputs/`: User-provided input files
  - `outputs/`: Generated analysis outputs
  - `processed/`: Intermediate processed data
  - `raw/`: Raw data from external sources
- **Format**: JSON, CSV, text files
- **Access**: Local filesystem access only

**Temporary Directory** (`temp/`)
- **Content**: Temporary processing files
- **Format**: Various formats
- **Access**: Local filesystem access only
- **Cleanup**: After report generation or manual deletion

#### 2. Environment Configuration

**`.env` File**
- **Content**: API keys, configuration settings
- **Location**: Project root directory
- **Access**: Local filesystem access only
- **Security**: Excluded from version control (`.gitignore`)
- **Backup**: User's responsibility

#### 3. In-Memory Storage

**Session State**
- **Content**: Workflow execution state
- **Location**: System memory (RAM)
- **Duration**: During workflow execution only
- **Persistence**: Not persisted to disk
- **Access**: System-internal only

### Data Retention Policy

#### Reports

- **Retention Period**: Indefinite (until manually deleted by user)
- **Deletion**: User-initiated manual deletion
- **Backup**: User's responsibility

#### Logs

- **Retention Period**: Configurable (default: indefinite)
- **Rotation**: Based on file size (configurable)
- **Deletion**: User-initiated manual deletion or automatic rotation
- **Archival**: User's responsibility

#### Temporary Files

- **Retention Period**: Until workflow completion or manual deletion
- **Automatic Cleanup**: Optional (configurable)
- **Deletion**: Automatic after report generation or manual deletion

#### Configuration Data

- **Retention Period**: Indefinite (until manually deleted by user)
- **Deletion**: User-initiated manual deletion
- **Security**: User's responsibility to protect API keys

### Data Backup

- **User Responsibility**: Users are responsible for backing up their data
- **System Backup**: System does not provide automatic backup services
- **Recommendations**: Regular backups of reports and important data

---

## Data Security

### Security Measures

#### 1. Access Control

**Local Filesystem**
- **Access**: Restricted to local filesystem only
- **Permissions**: Standard filesystem permissions
- **User Control**: Users control access through OS-level permissions

**API Keys**
- **Storage**: Stored in `.env` file (excluded from version control)
- **Access**: Read-only during system execution
- **Transmission**: Used only for API authentication (HTTPS)
- **Logging**: API keys are never logged or exposed

#### 2. Data Encryption

**At Rest**
- **Reports**: Stored as plain text (Markdown) or PDF (standard encryption)
- **Logs**: Stored as plain text
- **Configuration**: Stored as plain text in `.env` file
- **Recommendation**: Users should encrypt sensitive data if required

**In Transit**
- **API Communication**: All external API calls use HTTPS/TLS encryption
- **Data Transmission**: Secure protocols for all network communication

#### 3. Security Best Practices

**API Key Protection**
- API keys stored in `.env` file (not in code)
- `.env` file excluded from version control
- API keys never logged or exposed in error messages
- Users should never commit API keys to version control

**Log Security**
- Logs may contain sensitive information (prompts, responses)
- Users should protect log files appropriately
- Log rotation helps limit exposure

**File Permissions**
- Users should set appropriate file permissions
- Restrict access to sensitive directories
- Use OS-level security features

### Security Vulnerabilities

**Known Limitations**
- Local filesystem storage (no cloud security)
- Plain text storage for most data
- No built-in encryption for stored data
- User's responsibility for security

**Recommendations**
- Use encrypted filesystems for sensitive data
- Regularly update dependencies for security patches
- Monitor logs for suspicious activity
- Protect API keys and configuration files

### Incident Response

**Security Incidents**
- Users should report security incidents immediately
- Review logs for unauthorized access
- Rotate API keys if compromised
- Review and update security measures

---

## Data Sharing and Third-Party Services

### Third-Party Service Providers

The system integrates with the following third-party services:

#### 1. OpenAI

**Service**: Large Language Model (LLM) API  
**Purpose**: AI-powered analysis and report generation  
**Data Shared**: 
- Stock symbols
- Company information
- Analysis prompts
- Generated content

**Data Protection**: Subject to OpenAI's Privacy Policy  
**User Control**: Users can review OpenAI's privacy policy at https://openai.com/privacy  
**Data Retention**: Subject to OpenAI's data retention policies

#### 2. Yahoo Finance (yfinance)

**Service**: Stock market data API  
**Purpose**: Real-time and historical stock data retrieval  
**Data Shared**: 
- Stock symbols (public data)
- Data requests

**Data Protection**: Subject to Yahoo's Privacy Policy  
**User Control**: Users can review Yahoo's privacy policy  
**Data Retention**: Public data, no personal information shared

#### 3. DuckDuckGo

**Service**: Web search API  
**Purpose**: News and market intelligence search  
**Data Shared**: 
- Search queries (stock symbols, company names, sector terms)
- No personal information

**Data Protection**: Subject to DuckDuckGo's Privacy Policy  
**User Control**: DuckDuckGo does not track users  
**Data Retention**: Subject to DuckDuckGo's policies

### Data Sharing Principles

1. **Minimal Data Sharing**: Only necessary data is shared with third parties
2. **No Personal Information**: No PII is shared with third parties
3. **Public Data Only**: Only public stock market data is shared
4. **User Awareness**: Users should be aware of third-party data sharing

### Third-Party Privacy Policies

Users should review third-party privacy policies:

- **OpenAI**: https://openai.com/privacy
- **Yahoo**: https://policies.yahoo.com/us/en/yahoo/privacy/index.htm
- **DuckDuckGo**: https://duckduckgo.com/privacy

### Data Transfer

**International Transfers**
- Data may be transferred to third-party servers (OpenAI, Yahoo, DuckDuckGo)
- Transfer subject to third-party data protection policies
- Users should be aware of data location

**Safeguards**
- HTTPS/TLS encryption for all data transfers
- Secure API authentication
- No unencrypted data transmission

---

## User Rights

### Right to Access

**What You Can Access**
- All generated reports (stored in `reports/` directory)
- System logs (stored in `logs/` directory)
- Configuration data (stored in `.env` file)
- All data stored locally on your system

**How to Access**
- Direct filesystem access to stored files
- Review logs using text editors or log viewers
- Export reports in various formats

### Right to Rectification

**What You Can Correct**
- Input data (stock symbols, company names, sectors)
- Configuration settings
- Regenerate reports with corrected inputs

**How to Rectify**
- Modify input parameters and regenerate reports
- Update configuration in `.env` file
- Delete incorrect reports and regenerate

### Right to Erasure (Right to be Forgotten)

**What You Can Delete**
- Generated reports
- System logs
- Temporary files
- Configuration data

**How to Delete**
- Manual deletion of files from filesystem
- Delete reports from `reports/` directory
- Delete logs from `logs/` directory
- Delete temporary files from `temp/` directory
- Delete `.env` file (if desired)

**Note**: Data shared with third parties (OpenAI, Yahoo Finance, DuckDuckGo) is subject to their data retention policies and may not be immediately deletable.

### Right to Data Portability

**What You Can Export**
- Generated reports (PDF, Markdown formats)
- Analysis data (JSON, CSV formats)
- Logs (text format)

**How to Export**
- Copy files from storage directories
- Export reports in standard formats
- Extract data from logs

### Right to Object

**What You Can Object To**
- Data processing activities
- Third-party data sharing

**How to Object**
- Stop using the system
- Delete stored data
- Do not provide API keys (limits functionality)

**Note**: Some data processing is necessary for system functionality. Objecting may limit or prevent system usage.

### Right to Restrict Processing

**What You Can Restrict**
- Report generation
- Logging activities
- Data storage

**How to Restrict**
- Stop using the system
- Disable logging (if configurable)
- Delete stored data
- Do not provide API keys

### Right to Withdraw Consent

**How to Withdraw**
- Stop using the system
- Delete stored data
- Remove API keys
- Uninstall the system

**Note**: Withdrawing consent may prevent system usage.

### Exercising Your Rights

**Contact Information**
- See [Contact Information](#contact-information) section
- Report issues via GitHub Issues
- Contact maintainers for data-related requests

**Response Time**
- Requests will be addressed as soon as possible
- No formal SLA for open-source project
- Best effort response

---

## Data Deletion and Retention

### Automatic Deletion

**Temporary Files**
- Automatically deleted after report generation (if configured)
- Manual cleanup required if automatic cleanup disabled

**Session Data**
- Automatically cleared after workflow completion
- No persistent storage of session data

### Manual Deletion

**User-Initiated Deletion**
- Users can manually delete any stored data
- Delete reports, logs, temporary files
- Delete configuration files

**Deletion Methods**
- Filesystem deletion (standard OS commands)
- Delete directories: `reports/`, `logs/`, `temp/`, `data/`
- Delete individual files

### Data Retention After Deletion

**Local Storage**
- Deleted files are removed from filesystem
- Standard OS file deletion applies
- May be recoverable until overwritten (OS-dependent)

**Third-Party Services**
- Data shared with third parties subject to their retention policies
- OpenAI: Subject to their data retention policy
- Yahoo Finance: Public data, no personal information
- DuckDuckGo: Subject to their privacy policy

### Secure Deletion

**Recommendations**
- Use secure deletion tools for sensitive data
- Overwrite deleted files if required
- Use encrypted storage for sensitive data

---

## Compliance

### Applicable Regulations

This system may be subject to various data protection regulations depending on usage and jurisdiction:

#### General Data Protection Regulation (GDPR)

**Applicability**: If used by EU users or processes EU data

**Compliance Measures**:
- Data minimization principles
- User rights (access, rectification, erasure, portability)
- Transparency in data processing
- Security measures
- Data breach notification (user responsibility)

**GDPR Rights Covered**:
- Right to access
- Right to rectification
- Right to erasure
- Right to data portability
- Right to object
- Right to restrict processing

#### Other Regulations

**Applicability**: Depends on jurisdiction and usage

**Considerations**:
- Local data protection laws
- Industry-specific regulations
- International data transfer regulations

### Compliance Limitations

**Open-Source Project**
- No formal compliance certification
- Best-effort compliance measures
- User responsibility for compliance

**Third-Party Services**
- Compliance subject to third-party policies
- Users should review third-party compliance

### Compliance Recommendations

1. **Review Applicable Laws**: Understand local data protection requirements
2. **Assess Risk**: Evaluate data sensitivity and usage
3. **Implement Safeguards**: Use encryption, access controls, secure storage
4. **Monitor Compliance**: Regularly review data handling practices
5. **Update Policies**: Keep policies current with regulations

---

## Contact Information

### Data Protection Inquiries

**GitHub Repository**
- **URL**: [Repository URL]
- **Issues**: Create an issue for data-related questions
- **Discussions**: Use GitHub Discussions for general questions

**Maintainer Contact**
- **GitHub**: [@devendermishra]
- **Issues**: [GitHub Issues](https://github.com/devendermishra/stock-report-generator/issues)

### Reporting Data Incidents

**Security Incidents**
- Report via GitHub Issues (use security labels)
- Include details of the incident
- Provide steps to reproduce (if applicable)

**Data Breaches**
- Report immediately via GitHub Issues
- Include affected data types
- Describe potential impact

### General Inquiries

**Documentation**
- Check [Documentation](docs/) directory
- Review [README.md](README.md)
- See [DOCUMENTATION.md](docs/DOCUMENTATION.md)

**Support**
- Search [existing issues](https://github.com/devendermishra/stock-report-generator/issues)
- Create [new issue](https://github.com/devendermishra/stock-report-generator/issues/new)
- Join [discussions](https://github.com/devendermishra/stock-report-generator/discussions)

---

## Policy Updates

### Update Process

**Policy Versioning**
- Policies are versioned (current: 1.0)
- Major changes increment version number
- Minor updates may not change version

**Notification of Changes**
- Policy updates documented in this file
- "Last Updated" date reflects changes
- Users should review policy periodically

**Effective Date**
- Changes effective immediately upon update
- Continued use implies acceptance
- Users can stop using system if they disagree

### Update History

**Version 1.0** (Current)
- Initial policy documentation
- Comprehensive coverage of data handling
- User rights and compliance information

### Review Schedule

**Regular Reviews**
- Policy reviewed periodically
- Updates based on:
  - Regulatory changes
  - System changes
  - User feedback
  - Security incidents

**User Responsibility**
- Users should review policy updates
- Stay informed about changes
- Understand their rights and responsibilities

---

## Additional Resources

### Related Documentation

- [README.md](README.md) - Project overview and quick start
- [DOCUMENTATION.md](docs/DOCUMENTATION.md) - Comprehensive system documentation
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment and configuration guide
- [CODE_OF_CONDUCT.md](docs/CODE_OF_CONDUCT.md) - Code of conduct

### External Resources

- **GDPR Information**: https://gdpr.eu/
- **OpenAI Privacy Policy**: https://openai.com/privacy
- **Yahoo Privacy Policy**: https://policies.yahoo.com/us/en/yahoo/privacy/index.htm
- **DuckDuckGo Privacy Policy**: https://duckduckgo.com/privacy

### Best Practices

1. **Regular Backups**: Back up important reports and data
2. **API Key Security**: Protect API keys, never commit to version control
3. **Log Management**: Regularly review and rotate logs
4. **Data Cleanup**: Periodically clean temporary and old files
5. **Security Updates**: Keep dependencies updated for security patches
6. **Access Control**: Use appropriate file permissions
7. **Encryption**: Consider encryption for sensitive data

---

## Glossary

**API Key**: Authentication credential for external API services  
**GDPR**: General Data Protection Regulation (EU data protection law)  
**LLM**: Large Language Model (AI model for text generation)  
**NSE**: National Stock Exchange of India  
**PII**: Personally Identifiable Information  
**TLS**: Transport Layer Security (encryption protocol)  
**HTTPS**: Hypertext Transfer Protocol Secure (encrypted web protocol)

---

## Acknowledgment

By using the Stock Report Generator system, you acknowledge that you have read, understood, and agree to this Data Policy Documentation. You understand your rights regarding your data and the system's data handling practices.

**Important Reminders**:
- This project is for educational purposes only
- Do not use output as stock recommendations
- yfinance is for personal use only
- Do not use in production or commercial applications
- You are responsible for data security and compliance

---

**End of Data Policy Documentation**

