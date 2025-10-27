// Global storage for uploaded file data
let uploadedFileData = null;

// Enhanced file upload handling WITHOUT auto-analysis
function handleFileUpload(file) {
    console.log('🔍 [FRONTEND-DEBUG] handleFileUpload called');
    console.log('🔍 [FRONTEND-DEBUG] File:', file.name, 'Size:', file.size, 'Type:', file.type);
    
    const formData = new FormData();
    formData.append('file', file);

    showFileUploadProgress('file-input-area', file);
    
    // Clear previous upload state
    uploadedFileData = null;
    currentContent = null;
    
    // Clear any existing analysis results
    const analysisResults = document.getElementById('analysis-results');
    if (analysisResults) {
        analysisResults.innerHTML = '';
    }

    console.log('🔍 [FRONTEND-DEBUG] Sending upload request to /upload');
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('🔍 [FRONTEND-DEBUG] Upload response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('🔍 [FRONTEND-DEBUG] Upload response data:', data);
        
        if (data.success) {
            // Store uploaded file data globally
            uploadedFileData = {
                content: data.content,
                filename: data.filename,
                file_size: data.file_size,
                file_extension: data.file_extension,
                detected_format: data.detected_format,
                word_count: data.word_count,
                char_count: data.char_count
            };
            
            currentContent = data.content;
            
            console.log('✅ [FRONTEND-DEBUG] File uploaded successfully:', data.filename);
            console.log('📊 [FRONTEND-DEBUG] Format detected:', data.detected_format);
            console.log('📊 [FRONTEND-DEBUG] Content length:', data.content.length);
            console.log('📊 [FRONTEND-DEBUG] Content preview:', data.content.substring(0, 200));
            
            // Show success state (DO NOT analyze yet)
            showFileUploadSuccess(data);
            
            // Enable the Analyze button
            enableAnalyzeButton();
        } else {
            console.error('❌ [FRONTEND-DEBUG] Upload failed:', data.error);
            showError('file-input-area', data.error || 'Upload failed');
            disableAnalyzeButton();
        }
    })
    .catch(error => {
        console.error('❌ [FRONTEND-DEBUG] Upload error:', error);
        showError('file-input-area', 'Upload failed: ' + error.message);
        disableAnalyzeButton();
    });
}

// Enhanced file upload progress with PatternFly components
function showFileUploadProgress(elementId, file) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const fileSize = (file.size / 1024).toFixed(1);
    const fileType = file.type || 'Unknown';
    const fileExtension = file.name.split('.').pop().toUpperCase();
    
    element.innerHTML = `
        <div class="pf-v5-c-card app-card">
            <div class="pf-v5-c-card__header">
                <div class="pf-v5-c-card__header-main">
                    <h2 class="pf-v5-c-title pf-m-xl">
                        <i class="fas fa-upload pf-v5-u-mr-sm" style="color: var(--app-primary-color);"></i>
                        Processing Document
                    </h2>
                </div>
                <div class="pf-v5-c-card__actions">
                    <span class="pf-v5-c-label pf-m-blue">
                        <span class="pf-v5-c-label__content">
                            <i class="fas fa-spinner fa-spin pf-v5-c-label__icon"></i>
                            Uploading
                        </span>
                    </span>
                </div>
            </div>
            <div class="pf-v5-c-card__body">
                <!-- Progress Center -->
                <div class="pf-v5-c-empty-state pf-m-lg">
                    <div class="pf-v5-c-empty-state__content">
                        <div class="pf-v5-c-empty-state__icon">
                            <span class="pf-v5-c-spinner pf-m-xl pulse" role="status" aria-label="Processing">
                                <span class="pf-v5-c-spinner__clipper"></span>
                                <span class="pf-v5-c-spinner__lead-ball"></span>
                                <span class="pf-v5-c-spinner__tail-ball"></span>
                            </span>
                        </div>
                        <h2 class="pf-v5-c-title pf-m-lg">Extracting text from your document</h2>
                        <div class="pf-v5-c-empty-state__body">
                            <p class="pf-v5-u-mb-lg">Please wait while we process your ${fileExtension} file...</p>
                        </div>
                    </div>
                </div>
                
                <!-- File Information Grid -->
                <div class="pf-v5-l-grid pf-m-gutter pf-v5-u-mt-lg">
                    <div class="pf-v5-l-grid__item pf-m-6-col">
                        <div class="pf-v5-c-card pf-m-plain pf-m-bordered">
                            <div class="pf-v5-c-card__header">
                                <div class="pf-v5-c-card__header-main">
                                    <h3 class="pf-v5-c-title pf-m-md">
                                        <i class="fas fa-file pf-v5-u-mr-sm" style="color: var(--app-primary-color);"></i>
                                        File Information
                                    </h3>
                                </div>
                            </div>
                            <div class="pf-v5-c-card__body">
                                <dl class="pf-v5-c-description-list pf-m-horizontal">
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Name</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${file.name}</div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Size</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${fileSize} KB</div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Type</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${fileExtension}</div>
                                        </dd>
                                    </div>
                                </dl>
                            </div>
                        </div>
                    </div>
                    <div class="pf-v5-l-grid__item pf-m-6-col">
                        <div class="pf-v5-c-card pf-m-plain pf-m-bordered">
                            <div class="pf-v5-c-card__header">
                                <div class="pf-v5-c-card__header-main">
                                    <h3 class="pf-v5-c-title pf-m-md">
                                        <i class="fas fa-cogs pf-v5-u-mr-sm" style="color: var(--app-success-color);"></i>
                                        Processing Steps
                                    </h3>
                                </div>
                            </div>
                            <div class="pf-v5-c-card__body">
                                <div class="pf-v5-l-stack pf-m-gutter">
                                    <div class="pf-v5-l-stack__item">
                                        <div class="pf-v5-l-flex pf-m-align-items-center">
                                            <div class="pf-v5-l-flex__item">
                                                <i class="fas fa-check" style="color: var(--app-success-color);"></i>
                                            </div>
                                            <div class="pf-v5-l-flex__item pf-v5-u-ml-sm">
                                                <span class="pf-v5-u-font-size-sm">File validation</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="pf-v5-l-stack__item">
                                        <div class="pf-v5-l-flex pf-m-align-items-center">
                                            <div class="pf-v5-l-flex__item">
                                                <span class="pf-v5-c-spinner pf-m-sm" role="status">
                                                    <span class="pf-v5-c-spinner__clipper"></span>
                                                    <span class="pf-v5-c-spinner__lead-ball"></span>
                                                    <span class="pf-v5-c-spinner__tail-ball"></span>
                                                </span>
                                            </div>
                                            <div class="pf-v5-l-flex__item pf-v5-u-ml-sm">
                                                <span class="pf-v5-u-font-size-sm" style="color: var(--app-primary-color);">Text extraction</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="pf-v5-l-stack__item">
                                        <div class="pf-v5-l-flex pf-m-align-items-center">
                                            <div class="pf-v5-l-flex__item">
                                                <i class="fas fa-circle" style="color: var(--pf-v5-global--Color--200);"></i>
                                            </div>
                                            <div class="pf-v5-l-flex__item pf-v5-u-ml-sm">
                                                <span class="pf-v5-u-font-size-sm pf-v5-u-color-200">Content analysis</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="pf-v5-l-stack__item">
                                        <div class="pf-v5-l-flex pf-m-align-items-center">
                                            <div class="pf-v5-l-flex__item">
                                                <i class="fas fa-circle" style="color: var(--pf-v5-global--Color--200);"></i>
                                            </div>
                                            <div class="pf-v5-l-flex__item pf-v5-u-ml-sm">
                                                <span class="pf-v5-u-font-size-sm pf-v5-u-color-200">Results display</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Privacy Notice -->
                <div class="pf-v5-u-mt-lg">
                    <div class="pf-v5-c-alert pf-m-info pf-m-inline">
                        <div class="pf-v5-c-alert__icon">
                            <i class="fas fa-shield-alt"></i>
                        </div>
                        <h4 class="pf-v5-c-alert__title">Privacy Protected</h4>
                        <div class="pf-v5-c-alert__description">
                            Your document is processed securely and never stored on our servers. All analysis happens in real-time.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Show file upload success state
function showFileUploadSuccess(data) {
    const element = document.getElementById('file-input-area');
    if (!element) return;
    
    const fileSizeKB = (data.file_size / 1024).toFixed(1);
    const fileSizeMB = (data.file_size / (1024 * 1024)).toFixed(2);
    const displaySize = data.file_size > 1024 * 1024 ? `${fileSizeMB} MB` : `${fileSizeKB} KB`;
    
    // Format mapping for display
    const formatDisplayNames = {
        'asciidoc': 'AsciiDoc',
        'markdown': 'Markdown',
        'dita': 'DITA',
        'plaintext': 'Plain Text',
        'auto': 'Auto-detected'
    };
    const formatDisplay = formatDisplayNames[data.detected_format] || data.detected_format;
    
    element.innerHTML = `
        <div class="pf-v5-c-card app-card file-upload-success">
            <div class="pf-v5-c-card__header">
                <div class="pf-v5-c-card__header-main">
                    <h2 class="pf-v5-c-title pf-m-xl">
                        <i class="fas fa-check-circle pf-v5-u-mr-sm" style="color: var(--app-success-color);"></i>
                        File Uploaded Successfully
                    </h2>
                </div>
                <div class="pf-v5-c-card__actions">
                    <button class="pf-v5-c-button pf-m-plain" onclick="clearUploadedFile()" title="Remove file and upload another">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            <div class="pf-v5-c-card__body">
                <!-- Success Message -->
                <div class="pf-v5-c-alert pf-m-success pf-m-inline pf-v5-u-mb-md">
                    <div class="pf-v5-c-alert__icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <h4 class="pf-v5-c-alert__title">Ready for Analysis</h4>
                    <div class="pf-v5-c-alert__description">
                        ${data.message || 'Your file has been processed and is ready for analysis.'}
                    </div>
                </div>
                
                <!-- File Information Grid -->
                <div class="pf-v5-l-grid pf-m-gutter">
                    <div class="pf-v5-l-grid__item pf-m-12-col pf-m-6-col-on-md">
                        <div class="pf-v5-c-card pf-m-plain pf-m-bordered">
                            <div class="pf-v5-c-card__header">
                                <div class="pf-v5-c-card__header-main">
                                    <h3 class="pf-v5-c-title pf-m-md">
                                        <i class="fas fa-file-alt pf-v5-u-mr-sm" style="color: var(--app-primary-color);"></i>
                                        File Details
                                    </h3>
                                </div>
                            </div>
                            <div class="pf-v5-c-card__body">
                                <dl class="pf-v5-c-description-list pf-m-horizontal pf-m-compact">
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Filename</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text"><strong>${data.filename}</strong></div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">File Size</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${displaySize}</div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Format</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">
                                                <span class="pf-v5-c-label pf-m-blue">
                                                    <span class="pf-v5-c-label__content">${formatDisplay}</span>
                                                </span>
                                            </div>
                                        </dd>
                                    </div>
                                </dl>
                            </div>
                        </div>
                    </div>
                    <div class="pf-v5-l-grid__item pf-m-12-col pf-m-6-col-on-md">
                        <div class="pf-v5-c-card pf-m-plain pf-m-bordered">
                            <div class="pf-v5-c-card__header">
                                <div class="pf-v5-c-card__header-main">
                                    <h3 class="pf-v5-c-title pf-m-md">
                                        <i class="fas fa-chart-bar pf-v5-u-mr-sm" style="color: var(--app-success-color);"></i>
                                        Content Statistics
                                    </h3>
                                </div>
                            </div>
                            <div class="pf-v5-c-card__body">
                                <dl class="pf-v5-c-description-list pf-m-horizontal pf-m-compact">
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Words</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${data.word_count.toLocaleString()}</div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Characters</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">${data.char_count.toLocaleString()}</div>
                                        </dd>
                                    </div>
                                    <div class="pf-v5-c-description-list__group">
                                        <dt class="pf-v5-c-description-list__term">
                                            <span class="pf-v5-c-description-list__text">Status</span>
                                        </dt>
                                        <dd class="pf-v5-c-description-list__description">
                                            <div class="pf-v5-c-description-list__text">
                                                <span class="pf-v5-c-label pf-m-green">
                                                    <span class="pf-v5-c-label__content">
                                                        <i class="fas fa-check pf-v5-c-label__icon"></i>
                                                        Ready
                                                    </span>
                                                </span>
                                            </div>
                                        </dd>
                                    </div>
                                </dl>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Next Steps -->
                <div class="pf-v5-u-mt-md">
                    <div class="pf-v5-c-alert pf-m-info pf-m-inline">
                        <div class="pf-v5-c-alert__icon">
                            <i class="fas fa-info-circle"></i>
                        </div>
                        <h4 class="pf-v5-c-alert__title">Next Steps</h4>
                        <div class="pf-v5-c-alert__description">
                            Select a <strong>Content Type</strong> below, then click <strong>"Analyze Content"</strong> to begin the analysis.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Enable the Analyze button
function enableAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        console.log('✅ Analyze button enabled');
    }
}

// Disable the Analyze button
function disableAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        console.log('❌ Analyze button disabled');
    }
}

// Clear uploaded file and reset state
function clearUploadedFile() {
    uploadedFileData = null;
    currentContent = null;
    
    console.log('🗑️ [CLEAR-DEBUG] Clearing uploaded file...');
    
    // Switch back to text input tab
    const textTab = document.querySelector('[data-method="text"]');
    if (textTab) {
        textTab.click();
        console.log('✅ [CLEAR-DEBUG] Switched to text tab');
    }
    
    // Reset file input value
    const currentFileInput = document.getElementById('file-input');
    if (currentFileInput) {
        currentFileInput.value = '';
        console.log('✅ [CLEAR-DEBUG] Cleared file input value');
    }
    
    // Restore the original upload interface
    const fileInputArea = document.getElementById('file-input-area');
    console.log('🔍 [CLEAR-DEBUG] fileInputArea element:', fileInputArea);
    
    if (fileInputArea) {
        fileInputArea.innerHTML = `
            <input type="file" id="file-input" class="hidden-input" accept=".pdf,.docx,.md,.adoc,.dita,.xml,.txt">
            <div class="upload-interface upload-zone" id="upload-area">
                <div class="upload-content">
                    <div class="upload-icon-wrapper">
                        <i class="fas fa-cloud-upload-alt upload-icon"></i>
                    </div>
                    <h3 class="upload-title app-title-sm">Drop your document here</h3>
                    <p class="upload-description app-text-body">
                        or <button class="upload-link" onclick="document.getElementById('file-input').click()">browse files</button>
                    </p>
                    <div class="supported-formats">
                        <div class="formats-list">
                            <span class="format-item">PDF</span>
                            <span class="format-item">DOCX</span>
                            <span class="format-item">Markdown</span>
                            <span class="format-item">AsciiDoc</span>
                            <span class="format-item">DITA</span>
                            <span class="format-item">XML</span>
                        </div>
                        <p class="size-limit app-text-xs">Maximum file size: 16MB</p>
                    </div>
                </div>
            </div>
        `;
        
        console.log('✅ [CLEAR-DEBUG] Restored upload interface HTML');
        
        // Re-attach event handlers to the new elements
        const newFileInput = document.getElementById('file-input');
        const newUploadArea = document.getElementById('upload-area');
        
        console.log('🔍 [CLEAR-DEBUG] newFileInput:', newFileInput);
        console.log('🔍 [CLEAR-DEBUG] newUploadArea:', newUploadArea);
        
        if (newFileInput) {
            newFileInput.addEventListener('change', (e) => {
                console.log('📁 [CLEAR-DEBUG] File input changed');
                if (e.target.files.length > 0) {
                    processFileUpload(e.target.files[0]);
                }
            });
            console.log('✅ [CLEAR-DEBUG] Attached file input change listener');
        }
        
        if (newUploadArea) {
            // Drag and drop handlers
            newUploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                newUploadArea.classList.add('drag-over');
            });
            
            newUploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                if (!newUploadArea.contains(e.relatedTarget)) {
                    newUploadArea.classList.remove('drag-over');
                }
            });
            
            newUploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                newUploadArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    processFileUpload(files[0]);
                }
            });
            
            newUploadArea.addEventListener('click', (e) => {
                // Check if we're clicking on the upload link button
                if (e.target.classList.contains('upload-link')) {
                    return; // Let the onclick handle it
                }
                // Otherwise, trigger file input click
                if (newFileInput) {
                    newFileInput.click();
                }
            });
            
            console.log('✅ [CLEAR-DEBUG] Attached drag/drop listeners');
        }
    }
    
    // Clear analysis results
    const analysisResults = document.getElementById('analysis-results');
    if (analysisResults) {
        analysisResults.innerHTML = '';
        console.log('✅ [CLEAR-DEBUG] Cleared analysis results');
    }
    
    // Disable analyze button
    disableAnalyzeButton();
    
    console.log('✅ [CLEAR-DEBUG] File cleared and upload interface restored');
}

// Enhanced content analysis with better progress tracking
function analyzeContent(content, formatHint = 'auto', contentType = 'concept') {
    console.log('🔍 [FRONTEND-DEBUG] analyzeContent called');
    console.log('🔍 [FRONTEND-DEBUG] formatHint:', formatHint);
    console.log('🔍 [FRONTEND-DEBUG] contentType:', contentType);
    console.log('🔍 [FRONTEND-DEBUG] content length:', content.length);
    console.log('🔍 [FRONTEND-DEBUG] content preview:', content.substring(0, 200));
    
    // CRITICAL FIX: Ensure session ID exists and join WebSocket room
    if (!sessionId || !window.sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        window.sessionId = sessionId;
        console.log('🆕 [SESSION] Generated new session ID:', sessionId);
    }
    
    // CRITICAL: Join the WebSocket room BEFORE sending analysis request
    if (socket && socket.connected) {
        socket.emit('join_session', { session_id: sessionId });
        console.log('📡 [WEBSOCKET] Joined session room:', sessionId);
    } else {
        console.warn('⚠️ [WEBSOCKET] Socket not connected, progress updates may not work');
    }
    
    const analysisStartTime = performance.now(); // Track client-side timing
    
    // DISABLE analyze button and show spinner during analysis
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="pf-v5-c-spinner pf-m-sm" style="margin-right: 0.5rem;"><span class="pf-v5-c-spinner__clipper"></span><span class="pf-v5-c-spinner__lead-ball"></span><span class="pf-v5-c-spinner__tail-ball"></span></span> Analyzing...';
        console.log('🔒 Analyze button disabled during analysis');
    }
    
    showLoading('analysis-results', 'Starting comprehensive analysis...');

    const requestPayload = { 
        content: content,
        format_hint: formatHint,
        content_type: contentType,
        session_id: sessionId 
    };
    
    console.log('🔍 [FRONTEND-DEBUG] Sending analyze request with payload:', {
        content_length: content.length,
        format_hint: formatHint,
        content_type: contentType,
        session_id: sessionId
    });

    // Create an AbortController with extended timeout (5 minutes for long analysis)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        controller.abort();
        console.error('❌ [TIMEOUT] Analysis request timed out after 5 minutes');
    }, 300000); // 5 minutes = 300,000 milliseconds

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId); // Clear timeout on successful response
        console.log('🔍 [FRONTEND-DEBUG] Analyze response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('🔍 [FRONTEND-DEBUG] Analyze response data keys:', Object.keys(data));
        console.log('🔍 [FRONTEND-DEBUG] Analysis success:', data.success);
        if (data.structural_blocks) {
            console.log('🔍 [FRONTEND-DEBUG] Structural blocks count:', data.structural_blocks.length);
        }
        if (data.success) {
            const analysisEndTime = performance.now();
            const clientProcessingTime = (analysisEndTime - analysisStartTime) / 1000; // Convert to seconds
            
            // Store analysis data globally
            currentAnalysis = data.analysis;
            
            // Add timing and content type information
            currentAnalysis.client_processing_time = clientProcessingTime;
            currentAnalysis.server_processing_time = data.processing_time || data.analysis.processing_time;
            currentAnalysis.content_type = contentType;  // Store content type
            currentAnalysis.total_round_trip_time = clientProcessingTime;
            
            // 🎯 HYBRID APPROACH: Minimal delay to show remaining messages without blocking results
            const queueLength = window.progressUpdateQueue ? window.progressUpdateQueue.length : 0;
            const estimatedQueueTime = queueLength * 500; // 500ms per remaining message (fast but visible)
            
            // Only delay if queue has items, otherwise show immediately
            const displayDelay = queueLength > 0 ? estimatedQueueTime + 300 : 0;
            
            if (displayDelay > 0) {
                console.log(`🎬 Analysis complete! Brief ${displayDelay}ms wait for ${queueLength} queued messages (real-time approach)`);
            } else {
                console.log(`🎬 Analysis complete! No queue - showing results immediately`);
            }
            
            setTimeout(() => {
                // 🔧 CRITICAL FIX: Reset filters to show all issues when new analysis completes
                if (window.SmartFilterSystem) {
                    window.SmartFilterSystem.resetFilters();
                    console.log('🔄 Filters reset for new analysis');
                }
                
                // Display results
                const structuralBlocks = data.structural_blocks || null;
                
                // 🔧 FIX: Pass full data object so metadata_assistant is available
                const analysisWithMetadata = {
                    ...data.analysis,
                    metadata_assistant: data.metadata_assistant,
                    content_type: data.content_type
                };
                displayAnalysisResults(analysisWithMetadata, content, structuralBlocks);
                
                // Clear progress timers
                if (typeof clearProgressTimers === 'function') {
                    clearProgressTimers();
                }
                
                // Clear progress queue
                if (window.progressUpdateQueue) {
                    window.progressUpdateQueue = [];
                }
                
                // RE-ENABLE analyze button
                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Content';
                    console.log('🔓 Analyze button re-enabled');
                }
                
                // SHOW completion notification
                if (typeof showNotification === 'function') {
                    const errorCount = data.analysis?.errors?.length || 0;
                    showNotification(
                        `✅ Analysis complete! Found ${errorCount} issue${errorCount !== 1 ? 's' : ''} in ${data.processing_time?.toFixed(1)}s`, 
                        'success'
                    );
                }
                
                // Log performance metrics for debugging
                console.log('Analysis Performance:', {
                    server_time: currentAnalysis.server_processing_time,
                    client_time: clientProcessingTime,
                    content_type: contentType,
                    total_time: clientProcessingTime
                });
            }, displayDelay); // Minimal delay only if needed
        } else {
            showError('analysis-results', data.error || 'Analysis failed');
            
            // RE-ENABLE button on error
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Content';
            }
        }
    })
    .catch(error => {
        clearTimeout(timeoutId); // Clear timeout on error
        console.error('Analysis error:', error);
        
        let errorMessage = 'Analysis failed: ' + error.message;
        if (error.name === 'AbortError') {
            errorMessage = 'Analysis timed out after 5 minutes. Please try with a smaller document or contact support if the issue persists.';
        }
        
        showError('analysis-results', errorMessage);
        
        // RE-ENABLE button on error
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Content';
        }
        
        // Clear progress timers
        if (typeof clearProgressTimers === 'function') {
            clearProgressTimers();
        }
    });
} 