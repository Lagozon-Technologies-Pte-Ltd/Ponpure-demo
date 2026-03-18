// Global Variables
let tableName;
let clientTableData = {};
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let originalButtonHTML = "";
let isSidebarOpen = false;
let table_data = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Loaded - Initializing...");
    initPage();
});

async function initPage() {
    console.log("Page Initializing...");
    
    try {
        // Initialize UI components
        initUserProfile();
        initSidebarToggle(); // Only call this once
        
        // Load sessions first
        await loadSessions();
        
        // Initialize database connection if saved
        const savedDatabase = sessionStorage.getItem('selectedDatabase');
        const savedSection = sessionStorage.getItem('selectedSection');
        
        if (savedDatabase) {
            document.getElementById('database-dropdown').value = savedDatabase;
            connectToDatabase(savedDatabase);
            
            if (savedSection) {
                document.getElementById('section-dropdown').value = savedSection;
                fetchQuestions(savedSection);
            }
        }
        
        // Set default tab
        setTimeout(() => {
            document.getElementById('viewData').style.display = 'block';
            document.getElementById('viewData').classList.add('active');
        }, 100);
        
        // Attach event listeners
        attachEventListeners();
        
        console.log("Page initialized successfully");
    } catch (error) {
        console.error("Error during initialization:", error);
    }
}

// User Profile Functions
function initUserProfile() {
    console.log("Initializing user profile...");
    const userIcon = document.getElementById('user-profile-icon');
    const userDropdown = document.getElementById('user-dropdown');
    
    if (!userIcon) {
        console.error("User icon not found!");
        return;
    }
    
    userIcon.addEventListener('click', function(e) {
        e.stopPropagation();
        console.log("User icon clicked");
        userDropdown.classList.toggle('show');
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (userDropdown.classList.contains('show') && 
            !userIcon.contains(e.target) && 
            !userDropdown.contains(e.target)) {
            userDropdown.classList.remove('show');
        }
    });
}

// Sidebar Functions - FIXED VERSION
function initSidebarToggle() {
    console.log("Initializing sidebar toggle...");
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.getElementById('session-sidebar');
    
    if (!sidebarToggle || !sidebar) {
        console.error("Sidebar elements not found!");
        return;
    }
    
    // Create overlay if it doesn't exist - BUT ONLY FOR MOBILE
    if (!document.querySelector('.sidebar-overlay') && window.innerWidth < 768) {
        const overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
        
        // On mobile, clicking overlay closes sidebar
        overlay.addEventListener('click', closeSidebar);
    }
    
    sidebarToggle.addEventListener('click', function(e) {
        e.stopPropagation();
        console.log("Sidebar toggle clicked");
        toggleSidebar();
    });
    
    // On desktop, don't close when clicking outside
    // On mobile, we already have the overlay handler
    
    // Handle escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && isSidebarOpen) {
            closeSidebar();
        }
    });
}

function toggleSidebar() {
    isSidebarOpen = !isSidebarOpen;
    const sidebar = document.getElementById('session-sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    const sidebarToggleIcon = document.querySelector('.sidebar-toggle i');
    
    if (isSidebarOpen) {
        sidebar.classList.add('sidebar-open');
        document.body.classList.add('sidebar-open');
        
        // Only show overlay on mobile
        if (overlay && window.innerWidth < 768) {
            overlay.classList.add('active');
        }
        
        if (sidebarToggleIcon) sidebarToggleIcon.className = 'fas fa-times';
        console.log("Sidebar opened");
    } else {
        sidebar.classList.remove('sidebar-open');
        document.body.classList.remove('sidebar-open');
        
        // Hide overlay if it exists
        if (overlay) {
            overlay.classList.remove('active');
        }
        
        if (sidebarToggleIcon) sidebarToggleIcon.className = 'fas fa-bars';
        console.log("Sidebar closed");
    }
}

function closeSidebar() {
    if (isSidebarOpen) {
        isSidebarOpen = false;
        const sidebar = document.getElementById('session-sidebar');
        const overlay = document.querySelector('.sidebar-overlay');
        const sidebarToggleIcon = document.querySelector('.sidebar-toggle i');
        
        sidebar.classList.remove('sidebar-open');
        document.body.classList.remove('sidebar-open');
        
        if (overlay) {
            overlay.classList.remove('active');
        }
        
        if (sidebarToggleIcon) sidebarToggleIcon.className = 'fas fa-bars';
    }
}

// Database Functions
async function connectToDatabase(selectedDatabase) {
    console.log("Connecting to database:", selectedDatabase);
    const sectionDropdown = document.getElementById('section-dropdown');
    const connectionStatus = document.getElementById('connection-status');
    
    if (!sectionDropdown || !connectionStatus) {
        console.error("Required elements not found!");
        return;
    }
    
    // Clear previous options
    sectionDropdown.innerHTML = '<option value="" disabled selected><i class="fas fa-folder"></i> Select Subject</option>';
    
    // Update connection status
    connectionStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
    connectionStatus.classList.remove('connected');
    connectionStatus.classList.add('connecting');
    
    // Define sections based on database
    let sections = [];
    if (selectedDatabase === 'GCP') {
        sections = ['Demo', 'Mahindra-PoC-V2'];
    } else if (selectedDatabase === 'PostgreSQL-Azure') {
        sections = ['Mah-POC-Azure'];
    } else if (selectedDatabase === 'Azure SQL') {
        sections = ['Azure-SQL-DB'];
    } else {
        console.error('Unknown database selected:', selectedDatabase);
        connectionStatus.innerHTML = '<i class="fas fa-times-circle"></i> Connection Failed';
        connectionStatus.classList.remove('connecting');
        return;
    }
    
    // Add sections to dropdown
    sections.forEach(section => {
        const option = document.createElement('option');
        option.value = section;
        option.textContent = section;
        sectionDropdown.appendChild(option);
    });
    
    // Enable the dropdown and update status
    sectionDropdown.disabled = false;
    connectionStatus.innerHTML = `<i class="fas fa-check-circle"></i> Connected to ${selectedDatabase}`;
    connectionStatus.classList.remove('connecting');
    connectionStatus.classList.add('connected');
    
    // Save selection
    sessionStorage.setItem('selectedDatabase', selectedDatabase);
}

// Event Listeners
function attachEventListeners() {
    console.log("Attaching event listeners...");
    
    // Enter key for chat input
    const chatInput = document.getElementById("chat_user_query");
    if (chatInput) {
        chatInput.addEventListener("keyup", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Section dropdown change
    const sectionDropdown = document.getElementById('section-dropdown');
    if (sectionDropdown) {
        sectionDropdown.addEventListener('change', function() {
            const selectedDatabase = document.getElementById('database-dropdown').value;
            const selectedSection = this.value;
            
            if (selectedDatabase && selectedSection) {
                sessionStorage.setItem('selectedSection', selectedSection);
                fetchQuestions(selectedSection);
            }
        });
    }
    
    // Question type radio buttons
    document.querySelectorAll('input[name="questionType"]').forEach(radio => {
        radio.addEventListener('change', handleQuestionTypeChange);
    });
    
    // Tab navigation
    const tabButtons = document.querySelectorAll('.tab button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const tabName = this.textContent.trim();
            openTab(e, tabName === 'Data' ? 'viewData' : 'createVisualizations');
        });
    });
    
    // Initialize first tab as active
    if (tabButtons.length > 0) {
        tabButtons[0].classList.add('active');
    }
    
    // Copy buttons in modals
    document.querySelectorAll('.copy-btn-popup').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetEl = document.getElementById(targetId);
            
            if (targetEl) {
                const text = targetEl.textContent.trim();
                if (text && text !== "No SQL query available") {
                    navigator.clipboard.writeText(text)
                        .then(() => showToast('Copied to clipboard!', 'success'))
                        .catch(err => console.error('Failed to copy:', err));
                }
            }
        });
    });
}

// Chart Functions
function loadTableColumns(columnNames) {
    console.log("Loading table columns:", columnNames);
    const xAxisDropdown = document.getElementById("x-axis-dropdown");
    const yAxisDropdown = document.getElementById("y-axis-dropdown");
    
    if (!xAxisDropdown || !yAxisDropdown) {
        console.error("Chart dropdowns not found!");
        return;
    }
    
    // Reset dropdown options
    xAxisDropdown.innerHTML = '<option value="" disabled selected>Select X-Axis</option>';
    yAxisDropdown.innerHTML = '<option value="" disabled selected>Select Y-Axis</option>';
    
    // Populate options
    columnNames.forEach((column) => {
        const xOption = document.createElement("option");
        const yOption = document.createElement("option");
        
        xOption.value = column;
        xOption.textContent = column;
        yOption.value = column;
        yOption.textContent = column;
        
        xAxisDropdown.appendChild(xOption);
        yAxisDropdown.appendChild(yOption);
    });
}

async function generateChart() {
    console.log("Generating chart...");
    const xAxisDropdown = document.getElementById("x-axis-dropdown");
    const yAxisDropdown = document.getElementById("y-axis-dropdown");
    const chartTypeDropdown = document.getElementById("chart-type-dropdown");
    const chartLoading = document.getElementById("chart-loading");
    const chartContainer = document.getElementById("chart-container");
    
    const xAxis = xAxisDropdown.value;
    const yAxis = yAxisDropdown.value;
    const chartType = chartTypeDropdown.value;
    
    if (!xAxis || !yAxis || !chartType) {
        showToast("Please select all required fields.", "warning");
        return;
    }
    
    try {
        chartLoading.style.display = "block";
        chartContainer.innerHTML = "";
        
        const response = await fetch("/generate-chart", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                x_axis: xAxis,
                y_axis: yAxis,
                chart_type: chartType,
                table_data: table_data["Table data"] || table_data
            }),
        });
        
        const data = await response.json();
        chartLoading.style.display = "none";
        
        if (response.ok && data.chart) {
            const chartDiv = document.createElement("div");
            chartDiv.style.width = "100%";
            chartDiv.style.height = "500px";
            chartContainer.appendChild(chartDiv);
            
            Plotly.newPlot(chartDiv, JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
            showToast("Chart generated successfully!", "success");
        } else {
            throw new Error(data.error || "Failed to generate chart.");
        }
    } catch (error) {
        console.error("Error generating chart:", error);
        chartLoading.style.display = "none";
        showToast(error.message, "error");
    }
}

// Tab Navigation
function openTab(evt, tabName) {
    console.log("Opening tab:", tabName);
    
    // Hide all tab content
    document.querySelectorAll(".tabcontent").forEach(tab => {
        tab.style.display = "none";
        tab.classList.remove("active");
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll(".tablinks").forEach(tab => {
        tab.classList.remove("active");
    });
    
    // Show the current tab
    const activeTab = document.getElementById(tabName);
    if (activeTab) {
        activeTab.style.display = "block";
        activeTab.classList.add("active");
    }
    
    // Set the button that opened the tab as active
    if (evt && evt.currentTarget) {
        evt.currentTarget.classList.add("active");
    }
}

// Session Management
async function createNewSession() {
    console.log("Creating new session...");
    try {
        const res = await fetch('/new-session', { method: 'POST' });
        if (!res.ok) throw new Error('Failed to create new session');
        
        // Reload sessions list
        await loadSessions();
        
        // Clear UI
        clearChatUI();
        
        showToast("New session created!", "success");
    } catch (error) {
        console.error("Error creating new session:", error);
        showToast("Failed to create new session", "error");
    }
}

function clearChatUI() {
    document.getElementById("chat-messages").innerHTML = "";
    document.getElementById("tables_container").innerHTML = "";
    document.getElementById("xlsx-btn").innerHTML = "";
    document.getElementById("suggested-questions-container").style.display = "none";
    document.getElementById("user_query_display").querySelector('span').textContent = "";
    document.getElementById("sql_query_display").innerHTML = "";
    document.getElementById("user_query_display").style.display = "none";
}

// Dev Mode Toggle
function toggleDevMode() {
    console.log("Toggling dev mode...");
    const devModeToggle = document.getElementById('devModeToggle');
    const xlsxbtn = document.getElementById('xlsx-btn');
    
    if (!xlsxbtn) {
        console.error("xlsx-btn container not found!");
        return;
    }
    
    if (devModeToggle.checked) {
        // Create buttons if they don't exist
        let interpBtn = document.getElementById('interpBtn');
        let langchainBtn = document.getElementById('langchainBtn');
        
        if (!interpBtn) {
            interpBtn = document.createElement('button');
            interpBtn.id = 'interpBtn';
            interpBtn.innerHTML = '<i class="fas fa-comment-alt"></i> Interpretation Prompt';
            interpBtn.className = 'action-btn';
            interpBtn.onclick = showinterPrompt;
            xlsxbtn.appendChild(interpBtn);
        }
        
        if (!langchainBtn) {
            langchainBtn = document.createElement('button');
            langchainBtn.id = 'langchainBtn';
            langchainBtn.innerHTML = '<i class="fas fa-terminal"></i> Query Prompt';
            langchainBtn.className = 'action-btn';
            langchainBtn.onclick = showLangPromptPopup;
            xlsxbtn.appendChild(langchainBtn);
        }
        
        showToast("Developer mode enabled", "success");
    } else {
        // Remove buttons if they exist
        const interpBtn = document.getElementById('interpBtn');
        const langchainBtn = document.getElementById('langchainBtn');
        
        if (interpBtn) interpBtn.remove();
        if (langchainBtn) langchainBtn.remove();
        showToast("Developer mode disabled", "info");
    }
}

// Main Message Sending Function
async function sendMessage() {
    console.log("Sending message...");
    const userQueryInput = document.getElementById("chat_user_query");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const queryResultsDiv = document.getElementById('query-results');
    const tablesContainer = document.getElementById("tables_container");
    const xlsxbtn = document.getElementById("xlsx-btn");
    
    let userMessage = userQueryInput.value.trim();
    if (!userMessage) {
        showToast("Please enter a query", "warning");
        return;
    }
    
    // Get selected database and section
    const selectedDatabase = document.getElementById('database-dropdown').value;
    const selectedSection = document.getElementById('section-dropdown').value;
    
    if (!selectedDatabase || !selectedSection) {
        showToast("Please select both a database and a subject area", "warning");
        return;
    }
    
    // Clear previous results
    tablesContainer.innerHTML = "";
    xlsxbtn.innerHTML = "";
    document.getElementById("suggested-questions-container").style.display = "none";
    document.getElementById("user_query_display").style.display = "none";
    
    // Append user message to chat
    appendMessage(userMessage, 'user');
    userQueryInput.value = "";
    
    // Show loading state
    typingIndicator.style.display = "flex";
    queryResultsDiv.style.display = "block";
    
    try {
        const formData = new FormData();
        formData.append('user_query', userMessage);
        formData.append('section', selectedSection);
        formData.append('database', selectedDatabase);
        
        console.log("Sending request to server...");
        const response = await fetch("/submit", { method: "POST", body: formData });
        const data = await response.json();
        console.log("Server response:", data);
        
        // Hide typing indicator
        typingIndicator.style.display = "none";
        
        if (!response.ok) {
            throw new Error(data.chat_response || "An error occurred");
        }
        
        // Store table data
        if (data.tables_data) {
            clientTableData = {};
            for (const tableName in data.tables_data) {
                if (data.tables_data[tableName] && data.tables_data[tableName]['Table data']) {
                    clientTableData[tableName] = data.tables_data[tableName]['Table data'];
                }
            }
            table_data = data.tables_data;
        }
        
        // Append AI response
        let botResponse = data.chat_response || "";
        let fullResponse = `<strong>LLM Interpretation:</strong> ${data.llm_response || "Not available"}<br><br>${botResponse}`;
        appendMessage(fullResponse, 'ai');
        
        // Display suggested questions if available
        if (data.suggested_questions && Array.isArray(data.suggested_questions) && data.suggested_questions.length > 0) {
            setTimeout(() => displaySuggestedQuestions(data.suggested_questions), 100);
        }
        
        // Update page content
        updatePageContent(data);
        
        // Load table columns for visualization
        if (data.tables && data.tables_data) {
            const rows = data.tables_data["Table data"] || [];
            if (rows.length > 0) {
                const columnNames = Object.keys(rows[0]);
                loadTableColumns(columnNames);
            }
        }
        
    } catch (error) {
        console.error("Error:", error);
        typingIndicator.style.display = "none";
        appendMessage(`<strong>Error:</strong> ${error.message}. Please try again.`, 'ai');
    }
}

// Helper Functions
function appendMessage(content, sender) {
    const chatMessages = document.getElementById("chat-messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showToast(message, type = "info") {
    const toast = document.getElementById("custom-toast");
    if (!toast) {
        console.error("Toast element not found!");
        return;
    }
    
    const colors = {
        success: "#4CAF50",
        error: "#f44336",
        warning: "#FF9800",
        info: "#2196F3"
    };
    
    toast.textContent = message;
    toast.style.backgroundColor = colors[type] || colors.info;
    toast.style.display = "block";
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        toast.style.display = "none";
    }, 3000);
}

// Pagination Functions
function setupClientPagination(tableName, fullData, currentPage, recordsPerPage) {
    console.log("Setting up pagination for", tableName, "with", fullData.length, "records");
    
    if (!fullData || !Array.isArray(fullData)) {
        console.error(`Invalid data for table ${tableName}`, fullData);
        return;
    }
    
    const totalRecords = fullData.length;
    const totalPages = Math.ceil(totalRecords / recordsPerPage);
    currentPage = Math.max(1, Math.min(currentPage, totalPages));
    
    renderTablePage(tableName, fullData, currentPage, recordsPerPage);
    updatePaginationLinks(tableName, currentPage, totalPages, recordsPerPage);
}

function changePage(tableName, pageNumber, recordsPerPage = 10) {
    console.log("Changing to page", pageNumber, "for table", tableName);
    const fullData = clientTableData[tableName];
    if (!fullData) {
        console.error("No data found for table:", tableName);
        return;
    }
    setupClientPagination(tableName, fullData, pageNumber, recordsPerPage);
}

function renderTablePage(tableName, fullData, pageNumber, recordsPerPage) {
    console.log("Rendering page", pageNumber, "of table", tableName);
    const startIndex = (pageNumber - 1) * recordsPerPage;
    const endIndex = startIndex + recordsPerPage;
    const pageData = fullData.slice(startIndex, endIndex);
    const tableHtml = generateTableHtml(pageData, pageNumber, recordsPerPage);
    
    const tableDiv = document.getElementById(`${tableName}_table`);
    if (tableDiv) {
        tableDiv.innerHTML = tableHtml;
    } else {
        console.error("Table div not found:", `${tableName}_table`);
    }
}

function generateTableHtml(data, pageNumber = 1, recordsPerPage = 10) {
    if (!data || data.length === 0) return '<div class="no-data">No data available</div>';
    
    let html = '<table class="data-table"><thead><tr><th>S.No</th>';
    Object.keys(data[0]).forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    const startSerial = (pageNumber - 1) * recordsPerPage + 1;
    data.forEach((row, index) => {
        html += `<tr><td>${startSerial + index}</td>`;
        Object.values(row).forEach(cell => {
            html += `<td>${cell !== null && cell !== undefined ? cell : 'N/A'}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

// Voice Recording
async function toggleRecording() {
    console.log("Toggling recording...");
    const micButton = document.getElementById("chat-mic-button");
    
    if (!isRecording) {
        // Start recording
        originalButtonHTML = micButton.innerHTML;
        micButton.innerHTML = '<i class="fas fa-stop"></i>';
        micButton.style.color = "#f44336";
        isRecording = true;
        audioChunks = [];
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                isRecording = false;
                micButton.innerHTML = originalButtonHTML;
                micButton.style.color = "";
                
                if (audioChunks.length === 0) {
                    showToast("No audio recorded", "warning");
                    return;
                }
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append("file", audioBlob, "recording.webm");
                
                try {
                    const response = await fetch("/transcribe-audio/", { method: "POST", body: formData });
                    const data = await response.json();
                    
                    if (data.transcription) {
                        document.getElementById("chat_user_query").value = data.transcription;
                        showToast("Audio transcribed successfully", "success");
                    } else {
                        showToast("Failed to transcribe audio", "error");
                    }
                } catch (error) {
                    console.error("Error transcribing audio:", error);
                    showToast("Transcription failed", "error");
                }
            };
            
            mediaRecorder.start();
            showToast("Recording started... Click again to stop", "info");
        } catch (error) {
            console.error("Microphone access denied:", error);
            isRecording = false;
            micButton.innerHTML = originalButtonHTML;
            showToast("Microphone access denied", "error");
        }
    } else {
        // Stop recording
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
    }
}

// Question Type Handler
async function handleQuestionTypeChange(event) {
    console.log("Changing question type to:", event.target.value);
    const questionType = event.target.value;
    const radioButtons = document.querySelectorAll('input[name="questionType"]');
    
    radioButtons.forEach(radio => radio.disabled = true);
    
    try {
        await fetch('/new-session', { method: 'POST' });
        await fetch('/set-question-type', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question_type: questionType })
        });
        
        clearChatUI();
        showToast(`Switched to ${questionType} mode!`, "success");
    } catch (error) {
        console.error("Error changing question type:", error);
        event.target.checked = false;
        const currentType = document.body.dataset.initialQuestionType || 'generic';
        document.querySelector(`input[name="questionType"][value="${currentType}"]`).checked = true;
        showToast("Could not change question type", "error");
    } finally {
        radioButtons.forEach(radio => radio.disabled = false);
    }
}

// Session Management
async function loadSessions() {
    try {
        console.log("Loading sessions...");
        const res = await fetch("/sessions");
        if (!res.ok) {
            console.error("Failed to fetch sessions");
            return;
        }
        
        const sessions = await res.json();
        console.log("Sessions loaded:", sessions);
        const list = document.getElementById("session-list");
        
        if (!list) {
            console.error("Session list container not found!");
            return;
        }
        
        list.innerHTML = "";
        
        // Add sessions to main list
        sessions.forEach((s, index) => {
            const sessionDiv = createSessionElement(s);
            list.appendChild(sessionDiv);
        });
        
        // Mark first session as active if none is active
        if (sessions.length > 0 && !document.querySelector('.session-item.active')) {
            const firstSession = document.querySelector('.session-item');
            if (firstSession) {
                firstSession.classList.add('active');
            }
        }
        
    } catch (error) {
        console.error("Error loading sessions:", error);
    }
}

// SINGLE createSessionElement function
function createSessionElement(session) {
    const div = document.createElement("div");
    div.className = "session-item";
    div.dataset.sessionId = session.session_id || session.id;
    div.dataset.sessionTitle = session.title || 'Untitled Session';
    
    // Truncate title if too long
    let displayTitle = session.title || 'Untitled Session';
    if (displayTitle.length > 40) {
        displayTitle = displayTitle.substring(0, 40) + '...';
    }
    
    div.innerHTML = `
        <i class="fas fa-comment"></i>
        <div class="session-content">
            <div class="session-title">${displayTitle}</div>
            <div class="session-time">${formatTime(session.timestamp || session.created_at || Date.now())}</div>
        </div>
    `;
    
    // Add click event listener
    div.addEventListener('click', async (e) => {
        e.stopPropagation();
        await loadSessionMessages(session.session_id || session.id);
    });
    
    return div;
}

function formatTime(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        return 'Recent';
    }
}

async function loadSessionMessages(sessionId) {
    console.log("Loading session messages for ID:", sessionId);
    
    try {
        // Show loading state
        const chatMessages = document.getElementById("chat-messages");
        chatMessages.innerHTML = '<div class="loading-message">Loading session...</div>';
        
        const res = await fetch(`/sessions/${sessionId}`);
        if (!res.ok) {
            throw new Error(`Failed to load session: ${res.status}`);
        }
        
        const data = await res.json();
        console.log("Session data loaded:", data);
        
        // Clear chat
        chatMessages.innerHTML = "";
        
        // Load messages
        if (data.messages && Array.isArray(data.messages)) {
            data.messages.forEach(m => {
                const sender = m.role === 'user' ? 'user' : 'ai';
                const content = m.content || m.message || '';
                const user_query = m.user_query || '';
                const sql_query = m.sql_query || '';
                appendMessage(content, sender, user_query, sql_query);
            });
        } else {
            appendMessage("No messages in this session.", 'ai');
        }
        
        // Update active session in sidebar
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeSession = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
        if (activeSession) {
            activeSession.classList.add('active');
        }
        
        // Close sidebar on mobile (but not on desktop)
        if (window.innerWidth < 768) {
            closeSidebar();
        }
        
        // Update page title with session name
        const sessionTitle = activeSession?.dataset.sessionTitle || 'Session';
        document.title = `${sessionTitle} - DBQuery Assistant`;
        
        showToast(`Session loaded: ${sessionTitle}`, "success");
        
        // Clear results panel
        clearResultsPanel();
        
    } catch (error) {
        console.error("Error loading session messages:", error);
        const chatMessages = document.getElementById("chat-messages");
        chatMessages.innerHTML = `<div class="error-message">Failed to load session: ${error.message}</div>`;
        showToast("Failed to load session", "error");
    }
}

function clearResultsPanel() {
    document.getElementById("tables_container").innerHTML = "";
    document.getElementById("xlsx-btn").innerHTML = "";
    document.getElementById("suggested-questions-container").style.display = "none";
    document.getElementById("user_query_display").querySelector('span').textContent = "";
    document.getElementById("sql_query_display").innerHTML = "";
    document.getElementById("user_query_display").style.display = "none";
}

// Suggested Questions
function displaySuggestedQuestions(questions) {
    console.log("Displaying suggested questions:", questions);
    const container = document.getElementById("suggested-questions-container");
    const grid = document.getElementById("suggested-questions");
    
    if (!container || !grid) {
        console.error("Suggested questions containers not found!");
        return;
    }
    
    if (!questions || !Array.isArray(questions) || questions.length === 0) {
        container.style.display = "none";
        return;
    }
    
grid.innerHTML = "";  
questions.forEach((question, index) => {  
    const questionBox = document.createElement("button");  
    questionBox.className = "suggested-question-box";  
    questionBox.innerHTML = `  
    ${question}  
    <div class="click-hint">Click to use this question</div>  
    `;  
    questionBox.onclick = () => useSuggestedQuestion(question);  
    questionBox.style.animationDelay = `${index * 0.1}s`;  
    grid.appendChild(questionBox);  
});  

container.style.display = "block";  
setTimeout(() => container.scrollIntoView({ behavior: "smooth", block: "near" }));
}

function useSuggestedQuestion(question) {
    const inputField = document.getElementById("chat_user_query");
    inputField.value = question;
    inputField.focus();
    
    inputField.style.backgroundColor = "#f1f8ff";
    inputField.style.borderColor = "#4285f4";
    
    setTimeout(() => {
        inputField.style.backgroundColor = "";
        inputField.style.borderColor = "";
    }, 1000);
    
    showToast("Question added to input!", "success");
}

// Question Fetching
async function fetchQuestions(selectedSection) {
    const questionDropdown = document.getElementById("faq-questions");
    if (!questionDropdown) return;
    
    questionDropdown.innerHTML = '';
    
    if (selectedSection) {
        try {
            const response = await fetch(`/get_questions?subject=${selectedSection}`);
            const data = await response.json();
            
            if (data.questions && data.questions.length > 0) {
                data.questions.forEach(question => {
                    const option = document.createElement("option");
                    option.value = question;
                    questionDropdown.appendChild(option);
                });
            }
        } catch (error) {
            console.error("Error fetching questions:", error);
        }
    }
}

// Update Page Content
function updatePageContent(data) {
    console.log("Updating page content with:", data);
    
    // Update query display
    const userQueryDisplay = document.getElementById("user_query_display");
    if (userQueryDisplay) {
        userQueryDisplay.querySelector('span').textContent = data.user_query || "";
    }
    
    // Update SQL query
    const sqlQueryContent = document.getElementById("sql-query-content");
    if (sqlQueryContent) {
        const formattedQuery = data.query
            ? data.query
                .replace(/FROM/g, '\nFROM')
                .replace(/WHERE/g, '\nWHERE')
                .replace(/INNER JOIN/g, '\nINNER JOIN')
                .replace(/LEFT JOIN/g, '\nLEFT JOIN')
                .replace(/RIGHT JOIN/g, '\nRIGHT JOIN')
                .replace(/FULL JOIN/g, '\nFULL JOIN')
                .replace(/GROUP BY/g, '\nGROUP BY')
                .replace(/ORDER BY/g, '\nORDER BY')
                .replace(/HAVING/g, '\nHAVING')
                .replace(/SELECT/g, '\nSELECT')
                .replace(/ON/g, '\nON')
            : "No SQL query available";
        sqlQueryContent.textContent = formattedQuery;
    }
    
    // Update prompts
    const interpPromptContent = document.getElementById("interp-prompt-content");
    if (interpPromptContent && data.interprompt) {
        interpPromptContent.textContent = data.interprompt;
    }
    
    const langPromptContent = document.getElementById("lang-prompt-content");
    if (langPromptContent && data.langprompt) {
        const langdata = data.langprompt?.match(/template='([\s\S]*?)'\)\),/);
        let promptText = langdata ? langdata[1] : data.langprompt || "Not available";
        promptText = promptText.replace(/\\n/g, '\n');
        langPromptContent.textContent = promptText;
        Prism.highlightElement(langPromptContent);
    }
    
    // Clear containers
    const tablesContainer = document.getElementById("tables_container");
    const xlsxbtn = document.getElementById("xlsx-btn");
    if (tablesContainer) tablesContainer.innerHTML = "";
    if (xlsxbtn) xlsxbtn.innerHTML = "";
    
    // Create action buttons in the #xlsx-btn container
    createActionButtons(data, xlsxbtn);
    
    // Handle table display
    if (data.tables && data.tables.length > 0) {
        displayTables(data, tablesContainer, xlsxbtn);
    } else {
        if (tablesContainer) {
            tablesContainer.innerHTML = `
                <div class="no-data-message">
                    <p>${data.chat_response || "No data available"}</p>
                </div>
            `;
        }
    }
}

function createActionButtons(data, container) {
    if (!container) return;
    
    // Create Show Description button
    const showDescBtn = createButton("Show Description", "toggle-query-btn", "fas fa-eye", 
        function() {
            const userQueryDisplay = document.getElementById("user_query_display");
            if (userQueryDisplay.style.display === "none" || userQueryDisplay.style.display === "") {
                userQueryDisplay.style.display = "block";
                this.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Description';
                this.classList.add('active');
            } else {
                userQueryDisplay.style.display = "none";
                this.innerHTML = '<i class="fas fa-eye"></i> Show Description';
                this.classList.remove('active');
            }
        });
    container.appendChild(showDescBtn);
    
    // Create SQL Query button
    const sqlQueryBtn = createButton("SQL Query", "view-sql-query-btn", "fas fa-code", 
        showSQLQueryPopup, !data.query || data.query === "No SQL query available");
    container.appendChild(sqlQueryBtn);
    
    // Create Add to FAQs button
    const addToFaqsBtn = createButton("Add to FAQs", "add-to-faqs-btn", "fas fa-plus", 
        () => addToFAQs(document.getElementById('section-dropdown')?.value),
        !data.user_query);
    container.appendChild(addToFaqsBtn);
    
    // Only create Download button if there's table data
    if (data.tables && data.tables.length > 0) {
        const downloadButton = createButton("Download Excel", "download-button-all", "fas fa-download", 
            () => downloadSpecificTable(data.tables_data));
        container.appendChild(downloadButton);
    }
}

function createButton(text, id, icon, onClick, disabled = false) {
    const button = document.createElement("button");
    button.id = id;
    button.innerHTML = `<i class="${icon}"></i> ${text}`;
    button.className = "action-btn";
    button.onclick = onClick;
    button.disabled = disabled;
    return button;
}

function displayTables(data, tablesContainer, xlsxbtn) {
    data.tables.forEach((table) => {
        if (data.tables_data && data.tables_data[table.table_name]) {
            clientTableData[table.table_name] = data.tables_data[table.table_name]['Table data'] ||
                data.tables_data[table.table_name];
        }
        
        const tableWrapper = document.createElement("div");
        tableWrapper.className = "table-wrapper";
        tableWrapper.innerHTML = `
            <div class="table-header">
                <h4><i class="fas fa-table"></i> ${table.table_name}</h4>
            </div>
            <div id="${table.table_name}_table">${table.table_html || ''}</div>
            <div id="${table.table_name}_pagination"></div>
            <div class="feedback-section">
                <button class="like-button" onclick="submitFeedback('${table.table_name}', 'like')">
                    <i class="fas fa-thumbs-up"></i> Like
                </button>
                <button class="dislike-button" onclick="submitFeedback('${table.table_name}', 'dislike')">
                    <i class="fas fa-thumbs-down"></i> Dislike
                </button>
                <span id="${table.table_name}_feedback_message"></span>
            </div>
        `;
        if (tablesContainer) tablesContainer.appendChild(tableWrapper);
        
        // Setup pagination
        if (clientTableData[table.table_name]) {
            setupClientPagination(
                table.table_name,
                clientTableData[table.table_name],
                table.pagination?.current_page || 1,
                table.pagination?.records_per_page || 10
            );
        }
    });
}

// Modal Functions
function showSQLQueryPopup() {
    console.log("Showing SQL query popup");
    const sqlQueryText = document.getElementById("sql-query-content")?.textContent;
    if (!sqlQueryText || sqlQueryText === "No SQL query available") {
        showToast("No SQL query available", "warning");
        return;
    }
    document.getElementById("sql-query-popup").style.display = "flex";
    Prism.highlightAll();
}

function closeSQLQueryPopup() {
    document.getElementById("sql-query-popup").style.display = "none";
}

function showLangPromptPopup() {
    console.log("Showing lang prompt popup");
    document.getElementById("lang-prompt-popup").style.display = "flex";
    Prism.highlightAll();
}

function closepromptPopup() {
    document.getElementById("lang-prompt-popup").style.display = "none";
}

function showinterPrompt() {
    console.log("Showing interpretation prompt popup");
    document.getElementById("interp-prompt-popup").style.display = "flex";
    Prism.highlightAll();
}

function closeinterpromptPopup() {
    document.getElementById("interp-prompt-popup").style.display = "none";
}

// FAQ Functions
async function addToFAQs(subject) {
    if (!subject) {
        showToast("Please select a subject first!", "warning");
        return;
    }
    
    let userQuery = document.querySelector("#user_query_display span")?.innerText;
    if (!userQuery || !userQuery.trim()) {
        showToast("No query available to add to FAQs!", "warning");
        return;
    }
    
    try {
        const response = await fetch(`/add_to_faqs?subject=${encodeURIComponent(subject)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: userQuery })
        });
        
        const data = await response.json();
        showToast(data.message, data.success ? "success" : "error");
    } catch (error) {
        console.error('Error:', error);
        showToast("Failed to add query to FAQs!", "error");
    }
}

// Download Functions
async function downloadSpecificTable(table_data) {
    try {
        const response = await fetch('/download-table', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                table_name: "DBQuery_data",
                table_data: table_data
            })
        });
        
        if (!response.ok) throw new Error('Network response was not ok');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `DBQuery_data.xlsx`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        
        showToast("Download started!", "success");
    } catch (error) {
        console.error('Download error:', error);
        showToast("Download failed!", "error");
    }
}

// Pagination Links
function updatePaginationLinks(tableName, currentPage, totalPages, recordsPerPage) {
    const paginationDiv = document.getElementById(`${tableName}_pagination`);
    if (!paginationDiv) {
        console.error("Pagination div not found:", `${tableName}_pagination`);
        return;
    }
    
    paginationDiv.innerHTML = "";
    
    if (totalPages <= 1) return;
    
    const paginationList = document.createElement("ul");
    paginationList.className = "pagination";
    
    // Previous Button
    if (currentPage > 1) {
        const prevLi = document.createElement("li");
        prevLi.className = "page-item";
        prevLi.innerHTML = `<a href="javascript:void(0);" class="page-link" onclick="changePage('${tableName}', ${currentPage - 1}, ${recordsPerPage})">« Prev</a>`;
        paginationList.appendChild(prevLi);
    }
    
    // Page Numbers
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) startPage = Math.max(1, endPage - 4);
    
    if (startPage > 1) {
        const firstLi = document.createElement("li");
        firstLi.className = "page-item";
        firstLi.innerHTML = `<a href="javascript:void(0);" class="page-link" onclick="changePage('${tableName}', 1, ${recordsPerPage})">1</a>`;
        paginationList.appendChild(firstLi);
        
        if (startPage > 2) {
            const dotsLi = document.createElement("li");
            dotsLi.className = "page-item disabled";
            dotsLi.innerHTML = '<span class="page-link">...</span>';
            paginationList.appendChild(dotsLi);
        }
    }
    
    for (let page = startPage; page <= endPage; page++) {
        const pageLi = document.createElement("li");
        pageLi.className = `page-item ${page === currentPage ? 'active' : ''}`;
        pageLi.innerHTML = `<a href="javascript:void(0);" class="page-link" onclick="changePage('${tableName}', ${page}, ${recordsPerPage})">${page}</a>`;
        paginationList.appendChild(pageLi);
    }
    
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            const dotsLi = document.createElement("li");
            dotsLi.className = "page-item disabled";
            dotsLi.innerHTML = '<span class="page-link">...</span>';
            paginationList.appendChild(dotsLi);
        }
        
        const lastLi = document.createElement("li");
        lastLi.className = "page-item";
        lastLi.innerHTML = `<a href="javascript:void(0);" class="page-link" onclick="changePage('${tableName}', ${totalPages}, ${recordsPerPage})">${totalPages}</a>`;
        paginationList.appendChild(lastLi);
    }
    
    // Next Button
    if (currentPage < totalPages) {
        const nextLi = document.createElement("li");
        nextLi.className = "page-item";
        nextLi.innerHTML = `<a href="javascript:void(0);" class="page-link" onclick="changePage('${tableName}', ${currentPage + 1}, ${recordsPerPage})">Next »</a>`;
        paginationList.appendChild(nextLi);
    }
    
    paginationDiv.appendChild(paginationList);
}

// Feedback Submission
async function submitFeedback(tableName, feedbackType) {
    const userQuery = document.getElementById("chat_user_query").value;
    const sqlQuery = document.getElementById("sql-query-content")?.textContent || "";
    
    try {
        const response = await fetch("/submit_feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                table_name: tableName,
                feedback_type: feedbackType,
                user_query: userQuery,
                sql_query: sqlQuery
            }),
        });
        
        const data = await response.json();
        const feedbackMessage = document.getElementById(`${tableName}_feedback_message`);
        if (feedbackMessage) {
            feedbackMessage.textContent = data.message;
            feedbackMessage.style.color = data.success ? 'green' : 'red';
        }
        
        if (data.success) {
            showToast("Feedback submitted!", "success");
        }
    } catch (error) {
        console.error("Error submitting feedback:", error);
        showToast("Failed to submit feedback", "error");
    }
}

// Make functions globally available
window.connectToDatabase = connectToDatabase;
window.fetchQuestions = fetchQuestions;
window.sendMessage = sendMessage;
window.toggleRecording = toggleRecording;
window.createNewSession = createNewSession;
window.toggleDevMode = toggleDevMode;
window.generateChart = generateChart;
window.openTab = openTab;
window.showSQLQueryPopup = showSQLQueryPopup;
window.closeSQLQueryPopup = closeSQLQueryPopup;
window.showLangPromptPopup = showLangPromptPopup;
window.closepromptPopup = closepromptPopup;
window.showinterPrompt = showinterPrompt;
window.closeinterpromptPopup = closeinterpromptPopup;
window.addToFAQs = addToFAQs;
window.downloadSpecificTable = downloadSpecificTable;
window.changePage = changePage;
window.submitFeedback = submitFeedback;
window.useSuggestedQuestion = useSuggestedQuestion;