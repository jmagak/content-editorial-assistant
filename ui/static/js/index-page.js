/**
 * Index page JavaScript functionality
 * Handles file uploads, text input, drag & drop, and notifications
 */

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const textInput = document.getElementById('text-input');

    // Initialize modern editor functionality
    initializeModernEditor();

    // Enhanced drag and drop functionality
    if (uploadArea && fileInput) {
        console.log('✅ [INIT] Setting up drag/drop and file upload handlers');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            if (!uploadArea.contains(e.relatedTarget)) {
                uploadArea.classList.remove('drag-over');
            }
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                console.log('📁 [INIT] File dropped:', files[0].name);
                processFileUpload(files[0]);
            }
        });

        // Click handler for upload area (to trigger file browser)
        uploadArea.addEventListener('click', (e) => {
            // Don't interfere with the "browse files" link
            if (e.target.classList && e.target.classList.contains('upload-link')) {
                return;
            }
            console.log('📁 [INIT] Upload area clicked, opening file browser');
            fileInput.click();
        });

        // Single unified file input change handler
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                console.log('📁 [INIT] File selected:', e.target.files[0].name);
                
                // Clear text input when file is selected
                if (textInput) {
                    textInput.value = '';
                    // Update word count and button state
                    const wordCount = document.getElementById('word-count');
                    const charCount = document.getElementById('char-count');
                    const analyzeBtn = document.getElementById('analyze-btn');
                    
                    if (wordCount) wordCount.textContent = '0 words';
                    if (charCount) charCount.textContent = '0 characters';
                    if (analyzeBtn) analyzeBtn.disabled = true;
                }
                
                // Process the file upload
                processFileUpload(e.target.files[0]);
                
                // Switch to file tab to show upload status
                const fileTab = document.querySelector('[data-method="file"]');
                if (fileTab) {
                    fileTab.click();
                }
            }
        });
        
        console.log('✅ [INIT] File upload handlers initialized successfully');
    } else {
        console.warn('⚠️ [INIT] uploadArea or fileInput not found on page');
    }

    // Enhanced text input functionality
    if (textInput) {
        // Auto-resize textarea
        textInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.max(this.scrollHeight, 300) + 'px';
        });

        // Clear file input when text is entered
        textInput.addEventListener('input', () => {
            if (fileInput) fileInput.value = '';
        });
    }
});

/**
 * Enhanced file upload processing
 * @param {File} file - The file to process
 */
function processFileUpload(file) {
    // Validate file size
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File too large. Maximum size is 16MB.', 'danger');
        return;
    }

    // Validate file type
    const allowedTypes = ['.pdf', '.docx', '.md', '.adoc', '.dita', '.xml', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(fileExtension)) {
        showNotification('Unsupported file type. Please use PDF, DOCX, Markdown, AsciiDoc, DITA, XML, or TXT files.', 'danger');
        return;
    }

    console.log('✅ [PROCESS] File validated:', file.name, 'Extension:', fileExtension);
    
    // Show loading state
    showNotification('Processing file: ' + file.name, 'info');
    
    // Process the file - check if handleFileUpload is defined
    if (typeof handleFileUpload === 'function') {
        console.log('✅ [PROCESS] Calling handleFileUpload()');
        handleFileUpload(file);
    } else {
        console.error('❌ [PROCESS] handleFileUpload function not found!');
        showNotification('Error: Upload handler not available', 'danger');
    }
}

/**
 * Enhanced sample text loading
 */
function loadSampleText() {
    const sampleText = `The implementation of the new system was facilitated by the team in order to optimize performance metrics. Due to the fact that the previous system was inefficient, a large number of users were experiencing difficulties. The decision was made by management to utilize advanced technologies for the purpose of enhancing user experience and improving overall system reliability.

This document will provide an overview of the new features and improvements that have been implemented. It is important to note that these changes will have a significant impact on the overall user experience and system performance.`;
    
    const textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.value = sampleText;
        textInput.style.height = 'auto';
        textInput.style.height = (textInput.scrollHeight) + 'px';
        textInput.focus();
        showNotification('Sample text loaded. Click "Analyze Text" to see the AI analysis.', 'success');
    }
}

/**
 * Enhanced notification system
 * @param {string} message - The notification message
 * @param {string} type - The notification type (info, success, danger, warning)
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `pf-v5-c-alert pf-m-${type} pf-m-inline fade-in`;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.maxWidth = '400px';
    
    notification.innerHTML = `
        <div class="pf-v5-c-alert__icon">
            <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        </div>
        <p class="pf-v5-c-alert__title">${message}</p>
        <div class="pf-v5-c-alert__action">
            <button class="pf-v5-c-button pf-m-plain" type="button" onclick="this.closest('.pf-v5-c-alert').remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

/**
 * Handle input method tab switching
 */
function initializeModernEditor() {
    const tabs = document.querySelectorAll('.input-tab');
    const areas = document.querySelectorAll('.input-area');
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const wordCount = document.getElementById('word-count');
    const charCount = document.getElementById('char-count');

    // Tab switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const method = tab.getAttribute('data-method');
            
            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Update areas
            areas.forEach(area => area.classList.remove('active'));
            
            const targetArea = document.getElementById(`${method}-input-area`);
            if (targetArea) {
                targetArea.classList.add('active');
            }
            
            // Update analyze button state
            updateAnalyzeButton();
        });
    });

    // Text input functionality
    if (textInput) {
        textInput.addEventListener('input', () => {
            updateWordCount();
            updateAnalyzeButton();
        });
        
        // Initial update
        updateWordCount();
    }

    function updateWordCount() {
        if (!textInput || !wordCount || !charCount) return;
        
        const text = textInput.value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        const chars = text.length;
        
        wordCount.textContent = `${words} words`;
        charCount.textContent = `${chars} characters`;
    }

    function updateAnalyzeButton() {
        if (!analyzeBtn) return;
        
        const textInput = document.getElementById('text-input');
        const hasText = textInput && textInput.value.trim().length > 0;
        
        analyzeBtn.disabled = !hasText;
    }
}

/**
 * Clear the editor content
 */
function clearEditor() {
    const textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.value = '';
        textInput.focus();
        
        // Update counters and button state
        const wordCount = document.getElementById('word-count');
        const charCount = document.getElementById('char-count');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        if (wordCount) wordCount.textContent = '0 words';
        if (charCount) charCount.textContent = '0 characters';
        if (analyzeBtn) analyzeBtn.disabled = true;
        
        showNotification('Editor cleared', 'info');
    }
    
    // Clear uploaded file data if present
    if (typeof clearUploadedFile === 'function') {
        clearUploadedFile();
    }
}

/**
 * Enhanced sample text loading for modern editor
 */
function loadSampleText() {
    const sampleText = `The implementation of the new system was facilitated by the team in order to optimize performance metrics. Due to the fact that the previous system was inefficient, a large number of users were experiencing difficulties. The decision was made by management to utilize advanced technologies for the purpose of enhancing user experience and improving overall system reliability.

This document will provide an overview of the new features and improvements that have been implemented. It is important to note that these changes will have a significant impact on the overall user experience and system performance.

In order to ensure successful deployment, we need to make sure that all team members are aware of the new procedures. The implementation team has worked tirelessly to develop a solution that meets the needs of our users while maintaining the highest standards of quality and reliability.`;
    
    const textInput = document.getElementById('text-input');
    if (textInput) {
        // Switch to text input tab first
        const textTab = document.querySelector('[data-method="text"]');
        if (textTab && !textTab.classList.contains('active')) {
            textTab.click();
        }
        
        textInput.value = sampleText;
        textInput.focus();
        
        // Trigger auto-resize
        textInput.style.height = 'auto';
        textInput.style.height = Math.max(textInput.scrollHeight, 300) + 'px';
        
        // Update counters and button state
        const words = sampleText.trim().split(/\s+/).length;
        const chars = sampleText.length;
        
        const wordCount = document.getElementById('word-count');
        const charCount = document.getElementById('char-count');
        const analyzeBtn = document.getElementById('analyze-btn');
        
        if (wordCount) wordCount.textContent = `${words} words`;
        if (charCount) charCount.textContent = `${chars} characters`;
        if (analyzeBtn) analyzeBtn.disabled = false;
        
        showNotification('Sample text loaded successfully', 'success');
    }
}

/**
 * Analyze content from modern editor
 */
function analyzeEditorContent() {
    // Get content type from dropdown
    const contentTypeSelect = document.getElementById('content-type-select');
    const contentType = contentTypeSelect ? contentTypeSelect.value : 'concept';
    
    // Check if we have uploaded file data or text input
    if (typeof uploadedFileData !== 'undefined' && uploadedFileData !== null) {
        // Use uploaded file data with detected format
        console.log('📊 Analyzing uploaded file:', uploadedFileData.filename);
        console.log('📋 Detected format:', uploadedFileData.detected_format);
        console.log('📝 Content type:', contentType);
        
        // Don't show notification here - it's shown in analyzeContent function
        
        // Call analysis with uploaded file content and detected format
        if (typeof analyzeContent === 'function') {
            analyzeContent(
                uploadedFileData.content,
                uploadedFileData.detected_format,  // Use detected format from upload
                contentType
            );
        } else {
            console.error('analyzeContent function not available');
            showNotification('Analysis function not available', 'danger');
        }
    } else {
        // Get content from modern editor or legacy text input
        const modernEditor = document.getElementById('modern-editor');
        const textInput = document.getElementById('text-input');
        
        let content = '';
        if (modernEditor && modernEditor.value.trim()) {
            content = modernEditor.value.trim();
        } else if (textInput && textInput.value.trim()) {
            content = textInput.value.trim();
        }
        
        if (content) {
            console.log('📊 Analyzing text input');
            console.log('📝 Content type:', contentType);
            
            // Don't show notification here - it's shown in analyzeContent function
            
            // Call the actual analysis function with content type
            handleDirectTextAnalysis(content, contentType);
        } else {
            showNotification('Please enter some text or upload a file first.', 'warning');
        }
    }
}

/**
 * Handle direct text analysis (legacy compatibility)
 */
function analyzeDirectText() {
    analyzeEditorContent();
}

/**
 * Handle direct text analysis processing
 */
function handleDirectTextAnalysis(text, contentType = 'concept') {
    // Call the analyze endpoint directly for text input with content type
    if (typeof analyzeContent === 'function') {
        analyzeContent(text, 'auto', contentType);  // Pass contentType parameter
    } else {
        console.error('analyzeContent function not available');
        showNotification('Analysis function not available', 'error');
    }
} 