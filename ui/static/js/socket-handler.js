// Initialize Socket.IO connection
function initializeSocket() {
    console.log('🔍 DEBUG: Initializing Socket.IO connection...');
    
    // Configure Socket.IO with extended timeouts for long-running operations
    socket = io({
        // Extended timeout for long-running analysis operations
        timeout: 300000, // 5 minutes in milliseconds
        
        // Reconnection settings
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        
        // Ping/pong settings to keep connection alive
        pingInterval: 25000, // 25 seconds
        pingTimeout: 60000,  // 1 minute
        
        // Transport settings
        transports: ['websocket', 'polling'],
        upgrade: true,
        
        // Connection settings
        autoConnect: true,
        forceNew: false
    });
    
    socket.on('connect', function() {
        console.log('✅ Connected to server');
        
        // Check both global variables for session ID
        const currentSessionId = window.sessionId || sessionId;
        if (currentSessionId) {
            console.log(`🔍 DEBUG: Joining session room: ${currentSessionId}`);
            socket.emit('join_session', { session_id: currentSessionId });
        }
    });
    
    socket.on('session_id', function(data) {
        sessionId = data.session_id;
        console.log('✅ Session ID received:', sessionId);
    });
    
    socket.on('session_joined', function(data) {
        console.log('✅ Joined session room:', data.session_id);
    });
    
    socket.on('session_left', function(data) {
        console.log('✅ Left session room:', data.session_id);
    });
    
    socket.on('progress_update', function(data) {
        console.log('%c📡 WEBSOCKET EVENT RECEIVED', 'background: #0066cc; color: white; font-size: 14px; padding: 4px 8px; border-radius: 3px;');
        console.log('   📊 Progress:', data.progress);
        console.log('   🏷️  Step:', data.step);
        console.log('   📝 Status:', data.status);
        handleProgressUpdate(data);
    });
    
    socket.on('process_complete', function(data) {
        console.log('📡 WebSocket: process_complete event received');
        handleProcessComplete(data);
    });
    
    // BLOCK-LEVEL REWRITING WEBSOCKET HANDLERS
    socket.on('block_processing_start', function(data) {
        handleBlockProcessingStart(data);
    });
    
    socket.on('station_progress_update', function(data) {
        handleStationProgressUpdate(data);
    });
    
    socket.on('block_processing_complete', function(data) {
        handleBlockProcessingComplete(data);
    });
    
    socket.on('block_processing_error', function(data) {
        handleBlockProcessingError(data);
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
    });
}

// Handle real-time progress updates with PatternFly components
function handleProgressUpdate(data) {
    console.log('\n🎯 ============ WEBSOCKET PROGRESS UPDATE RECEIVED ============');
    console.log('   📊 Progress:', data.progress + '%');
    console.log('   🏷️  Step:', data.step);
    console.log('   📝 Status:', data.status);
    console.log('   💬 Detail:', data.detail);
    console.log('   🆔 Session:', data.session_id);
    console.log('   ⏰ Timestamp:', new Date().toISOString());
    console.log('============================================================\n');
    
    // Update legacy status elements if they exist
    const statusElement = document.getElementById('current-status');
    const detailElement = document.getElementById('status-detail');
    
    if (statusElement && detailElement) {
        statusElement.textContent = data.status;
        detailElement.textContent = data.detail;
        console.log('   ✅ Updated legacy status elements');
    }
    
    // 🆕 UPDATE LIVE PROGRESS BAR (NEW WORLD-CLASS UI)
    updateLiveProgressBar(data);
    
    // Update step indicators based on actual progress
    console.log('   🔄 Updating step indicators...');
    updateStepIndicators(data.step, data.progress);
    
    // ENHANCED: Also update assembly line progress for block-level processing
    // Check if this is a pass-related update that should update assembly line progress
    if (data.step && (data.step.includes('Pass') || data.step.includes('station') || data.step.includes('Processing'))) {
        console.log('   🏭 This looks like an assembly line update');
        // Try to find the currently processing block and update its assembly line
        const currentProcessingBlock = window.blockRewriteState?.currentlyProcessingBlock;
        console.log('   📋 Current processing block:', currentProcessingBlock);
        if (currentProcessingBlock && data.progress) {
            const progressPercent = parseInt(data.progress) || 0;
            console.log('   📊 Updating assembly line progress:', progressPercent + '%');
            updateBlockAssemblyLineProgress(currentProcessingBlock, progressPercent, data.detail || data.status);
            console.log(`   ✅ Updated assembly line progress for block ${currentProcessingBlock}: ${progressPercent}%`);
        } else {
            console.log('   ⚠️  No current processing block or progress value');
        }
    } else {
        console.log('   ⚠️  This does not look like an assembly line update');
    }
    console.log('   ✅ Progress update handling complete\n');
}

// Queue for pending updates (to ensure minimum display time)
window.progressUpdateQueue = window.progressUpdateQueue || [];
window.isProcessingUpdate = window.isProcessingUpdate || false;
window.lastUpdateTime = window.lastUpdateTime || 0;

// Update live progress bar with MINIMUM DISPLAY TIME for each stage
function updateLiveProgressBar(data) {
    console.log('🔥 updateLiveProgressBar called with:', {
        step: data.step,
        status: data.status,
        detail: data.detail,
        progress: data.progress
    });
    
    // Get progress elements
    const progressIndicator = document.getElementById('live-progress-indicator');
    const progressPercentage = document.getElementById('live-progress-percentage');
    const progressStatus = document.getElementById('live-progress-status');
    const progressDetail = document.getElementById('live-progress-detail');
    const progressBarElement = document.getElementById('live-progress-bar-element');
    const stageIcon = document.getElementById('live-stage-icon');
    const stageText = document.getElementById('live-stage-text');
    
    console.log('🔍 Elements found:', {
        progressIndicator: !!progressIndicator,
        progressStatus: !!progressStatus,
        stageText: !!stageText
    });
    
    // If elements don't exist, wait briefly and retry once
    if (!progressIndicator || !progressPercentage) {
        console.log('   ⏳ Live progress elements not found yet, waiting for DOM...');
        
        // Retry after 100ms to give DOM time to render
        setTimeout(() => {
            const retryIndicator = document.getElementById('live-progress-indicator');
            const retryPercentage = document.getElementById('live-progress-percentage');
            
            if (retryIndicator && retryPercentage) {
                console.log('   ✅ Retry successful - elements now found, updating...');
                updateLiveProgressBar(data); // Recursive call with same data
            } else {
                console.log('   ⚠️  Live progress elements still not found (analysis may be complete)');
            }
        }, 100);
        return;
    }
    
    // Add to queue and process with minimum display time
    window.progressUpdateQueue.push(data);
    processProgressQueue();
}

// Process progress updates
function processProgressQueue() {
    if (window.isProcessingUpdate || window.progressUpdateQueue.length === 0) {
        return;
    }
    
    const MINIMUM_DISPLAY_TIME = 500; // 500ms minimum per stage (11 messages × 500ms = 5.5s total - fast but visible!)
    const timeSinceLastUpdate = Date.now() - window.lastUpdateTime;
    
    if (timeSinceLastUpdate < MINIMUM_DISPLAY_TIME && window.lastUpdateTime > 0) {
        // Wait for minimum display time before showing next update
        const waitTime = MINIMUM_DISPLAY_TIME - timeSinceLastUpdate;
        console.log(`⏱️  Waiting ${waitTime}ms before next update (${(waitTime/1000).toFixed(1)}s)`);
        setTimeout(processProgressQueue, waitTime);
        return;
    }
    
    // Get next update from queue
    const data = window.progressUpdateQueue.shift();
    window.isProcessingUpdate = true;
    
    console.log(`📺 DISPLAYING ON SCREEN: ${data.step} at ${data.progress}% (will show for ${MINIMUM_DISPLAY_TIME}ms)`);
    
    // Apply the update immediately
    applyProgressUpdate(data);
    
    // Mark update as complete and process next
    window.lastUpdateTime = Date.now();
    window.isProcessingUpdate = false;
    
    // Process next item in queue if available
    if (window.progressUpdateQueue.length > 0) {
        setTimeout(processProgressQueue, MINIMUM_DISPLAY_TIME);
    }
}

// Actually apply the progress update to the UI
function applyProgressUpdate(data) {
    const progressIndicator = document.getElementById('live-progress-indicator');
    const progressPercentage = document.getElementById('live-progress-percentage');
    const progressStatus = document.getElementById('live-progress-status');
    const progressDetail = document.getElementById('live-progress-detail');
    const progressBarElement = document.getElementById('live-progress-bar-element');
    const stageIcon = document.getElementById('live-stage-icon');
    const stageText = document.getElementById('live-stage-text');
    
    if (!progressIndicator || !progressPercentage) {
        return;
    }
    
    // Stop fallback animation since we're getting real updates
    if (window.fallbackProgressInterval) {
        clearInterval(window.fallbackProgressInterval);
        window.fallbackProgressInterval = null;
        console.log('   🛑 Stopped fallback animation - real WebSocket updates received');
    }
    if (window.progressAnimationInterval) {
        clearTimeout(window.progressAnimationInterval);
        window.progressAnimationInterval = null;
    }
    
    // Parse progress value (handle both number and string)
    const progress = parseInt(data.progress) || 0;
    
    const currentProgress = parseInt(progressBarElement?.getAttribute('aria-valuenow')) || 0;
    
    if (progress < currentProgress) {
        console.warn(`   ⚠️  Ignoring backwards progress: ${progress}% < ${currentProgress}%`);
        return; // Don't update if going backwards
    }
    
    console.log(`%c✅ Progress UPDATED: ${currentProgress}% → ${progress}%`, 'background: #3e8635; color: white; font-weight: bold; padding: 3px 6px;');
    
    // Update progress bar width with smooth animation
    progressIndicator.style.width = `${progress}%`;
    
    // Update percentage display
    progressPercentage.textContent = `${progress}%`;
    
    // Update aria-valuenow for accessibility
    if (progressBarElement) {
        progressBarElement.setAttribute('aria-valuenow', progress);
    }
    
    // Get LIVE stage-specific message
    const stageInfo = getStageInfo(data.step, data.status, data.detail, progress);
    console.log('%c📋 Stage Info', 'background: #0066cc; color: white; padding: 2px 6px;', stageInfo);
    
    // FORCE UPDATE - Update LIVE status message (main heading) with animation
    if (progressStatus) {
        const newStatus = stageInfo.title || data.status || 'Processing...';
        // Add fade effect for visibility
        progressStatus.style.opacity = '0.6';
        setTimeout(() => {
            progressStatus.textContent = newStatus;
            progressStatus.style.opacity = '1';
            progressStatus.style.transition = 'opacity 0.3s ease';
        }, 50);
        console.log(`%c✍️  STATUS: ${newStatus}`, 'color: #0066cc; font-weight: bold; font-size: 13px;');
    }
    
    // FORCE UPDATE - Update LIVE detail message (subtitle) with animation
    if (progressDetail) {
        const newDetail = stageInfo.detail || data.detail || 'Analyzing your content';
        progressDetail.style.opacity = '0.6';
        setTimeout(() => {
            progressDetail.textContent = newDetail;
            progressDetail.style.opacity = '1';
            progressDetail.style.transition = 'opacity 0.3s ease';
        }, 50);
        console.log(`   💬 Detail: ${newDetail}`);
    }
    
    // FORCE UPDATE - Update LIVE stage display with pulse animation
    if (stageIcon && stageText) {
        stageIcon.className = stageInfo.icon || 'fas fa-spinner fa-pulse';
        stageIcon.style.color = stageInfo.color || '#0066cc';
        const newStage = stageInfo.stage || 'Processing';
        // Add scale pulse effect
        stageText.style.transform = 'scale(1.05)';
        setTimeout(() => {
            stageText.textContent = newStage;
            stageText.style.color = stageInfo.color || '#0066cc';
            stageText.style.transform = 'scale(1)';
            stageText.style.transition = 'transform 0.2s ease, color 0.3s ease';
        }, 50);
        console.log(`   🏷️  Stage Badge: ${newStage}`);
    }
    
    // FORCE GRADIENT UPDATE - Change progress bar color/gradient based on progress
    if (progress >= 100) {
        progressIndicator.style.background = 'linear-gradient(90deg, #3e8635, #5ba352)'; // Green
        progressIndicator.style.boxShadow = '0 0 10px rgba(62, 134, 53, 0.5)'; // Green glow
        console.log('🟢 Applied GREEN gradient');
        // Clear all timers when complete
        if (typeof clearProgressTimers === 'function') {
            clearProgressTimers();
        }
    } else if (progress >= 75) {
        progressIndicator.style.background = 'linear-gradient(90deg, #0066cc, #004499)'; // Darker blue
        progressIndicator.style.boxShadow = '0 0 10px rgba(0, 102, 204, 0.4)'; // Blue glow
        console.log('🔵 Applied DARK BLUE gradient');
    } else {
        progressIndicator.style.background = 'linear-gradient(90deg, #0088ff, #0066cc)'; // Light blue
        progressIndicator.style.boxShadow = '0 0 10px rgba(0, 136, 255, 0.3)'; // Light blue glow
        console.log('🔵 Applied LIGHT BLUE gradient');
    }
    
    console.log(`   ✅ Live progress bar updated: ${progress}% - ${stageInfo.title}`);
}

// 🆕 Get stage-specific information for LIVE display
function getStageInfo(step, status, detail, progress) {
    const stageMap = {
        'analysis_start': {
            title: '🚀 Initializing Analysis',
            detail: 'Setting up analysis pipeline and preparing content',
            stage: 'Initialization',
            icon: 'fas fa-rocket',
            color: '#0066cc'
        },
        'structural_parsing': {
            title: '📄 Parsing Document Structure',
            detail: 'Analyzing document format and extracting content blocks',
            stage: 'Parsing Document',
            icon: 'fas fa-file-code',
            color: '#0066cc'
        },
        'style_analysis': {
            title: '✍️ Style Analysis Complete',
            detail: 'Finished checking grammar, style rules, and sentence structure',
            stage: 'Style Checked',
            icon: 'fas fa-spell-check',
            color: '#0066cc'
        },
        'compliance_check': {
            title: '✅ Validating Compliance',
            detail: 'Ensuring content meets modular documentation standards',
            stage: 'Compliance Check',
            icon: 'fas fa-shield-alt',
            color: '#0066cc'
        },
        'metadata_generation': {
            title: '🏷️ Generating Metadata',
            detail: 'Extracting title, keywords, and taxonomy classification',
            stage: 'Metadata Generation',
            icon: 'fas fa-tags',
            color: '#0066cc'
        },
        'metadata_start': {
            title: '🏷️ Starting Metadata Extraction',
            detail: 'Initializing AI-powered metadata generation',
            stage: 'Metadata Starting',
            icon: 'fas fa-cog fa-spin',
            color: '#0066cc'
        },
        'metadata_title': {
            title: '📝 Extracting Title',
            detail: detail || 'Analyzing content to extract the most relevant title',
            stage: 'Title Extraction',
            icon: 'fas fa-heading',
            color: '#0066cc'
        },
        'metadata_keywords': {
            title: '🔑 Extracting Keywords',
            detail: detail || 'Identifying key terms and concepts',
            stage: 'Keyword Extraction',
            icon: 'fas fa-key',
            color: '#0066cc'
        },
        'metadata_description': {
            title: '📋 Generating Description',
            detail: detail || 'Creating AI-powered content summary',
            stage: 'Description Generation',
            icon: 'fas fa-align-left',
            color: '#0066cc'
        },
        'metadata_taxonomy': {
            title: '🏷️ Classifying Taxonomy',
            detail: detail || 'Categorizing content using AI classification',
            stage: 'Taxonomy Classification',
            icon: 'fas fa-sitemap',
            color: '#0066cc'
        },
        'metadata_complete': {
            title: '🎉 Metadata Complete',
            detail: 'Successfully generated all metadata and classifications',
            stage: 'Metadata Generated',
            icon: 'fas fa-check-circle',
            color: '#3e8635'
        },
        'analysis_complete': {
            title: '✨ Analysis Complete!',
            detail: 'All checks finished - preparing your results',
            stage: 'Complete',
            icon: 'fas fa-check-circle',
            color: '#3e8635'
        }
    };
    
    // Return stage info or fallback
    return stageMap[step] || {
        title: status || 'Processing...',
        detail: detail || 'Analyzing your content',
        stage: 'Processing',
        icon: 'fas fa-spinner fa-pulse',
        color: '#06c'
    };
}

// Note: Old multi-stage indicator function removed - now using single live stage display

// Update step indicators using PatternFly progress components
function updateStepIndicators(currentStep, progress) {
    const stepMapping = {
        'analysis_start': 'step-analysis',
        'structural_parsing': 'step-analysis',
        'spacy_processing': 'step-analysis',
        'block_mapping': 'step-analysis',
        'metrics_calculation': 'step-analysis',
        'analysis_complete': 'step-analysis',
        'rewrite_start': 'step-analysis',
        'pass1_start': 'step-pass1',
        'pass1_processing': 'step-pass1',
        'pass1_complete': 'step-pass1',
        'pass2_start': 'step-pass2',
        'pass2_processing': 'step-pass2',
        'validation': 'step-complete',
        'rewrite_complete': 'step-complete'
    };
    
    const targetStepId = stepMapping[currentStep];
    if (!targetStepId) return;
    
    // Mark previous steps as complete
    const allSteps = document.querySelectorAll('.step-item');
    let foundTarget = false;
    
    allSteps.forEach((step) => {
        const stepId = step.id;
        
        if (stepId === targetStepId) {
            foundTarget = true;
            // Mark current step as active
            step.classList.remove('completed', 'pf-m-success');
            step.classList.add('active', 'pf-m-info');
            const icon = step.querySelector('.step-icon');
            if (currentStep.includes('complete')) {
                // Step is complete
                step.classList.remove('active', 'pf-m-info');
                step.classList.add('completed', 'pf-m-success');
                icon.innerHTML = '<i class="fas fa-check-circle" style="color: var(--app-success-color);"></i>';
            } else {
                // Step is in progress
                icon.innerHTML = `
                    <span class="pf-v5-c-spinner pf-m-sm" role="status">
                        <span class="pf-v5-c-spinner__clipper"></span>
                        <span class="pf-v5-c-spinner__lead-ball"></span>
                        <span class="pf-v5-c-spinner__tail-ball"></span>
                    </span>
                `;
            }
        } else if (!foundTarget) {
            // Mark previous steps as complete
            step.classList.remove('active', 'pf-m-info');
            step.classList.add('completed', 'pf-m-success');
            const icon = step.querySelector('.step-icon');
            icon.innerHTML = '<i class="fas fa-check-circle" style="color: var(--app-success-color);"></i>';
        } else {
            // Mark future steps as pending
            step.classList.remove('active', 'completed', 'pf-m-info', 'pf-m-success');
            const icon = step.querySelector('.step-icon');
            icon.innerHTML = '<i class="fas fa-circle" style="color: var(--pf-v5-global--Color--200);"></i>';
        }
    });
}

// Handle process completion
function handleProcessComplete(data) {
    console.log('Process complete:', data);
    
    if (data.success && data.data) {
        // Display results based on the type of process
        if (data.data.analysis) {
            // Analysis completed
            currentAnalysis = data.data.analysis;
            const structuralBlocks = data.data.structural_blocks || null;
            
            // 🔧 CRITICAL FIX: Reset filters to show all issues when new analysis completes
            if (window.SmartFilterSystem) {
                window.SmartFilterSystem.resetFilters();
                console.log('🔄 Filters reset for new analysis');
            }
            
            // 🔧 FIX: Pass full data object so metadata_assistant is available
            const analysisWithMetadata = {
                ...data.data.analysis,
                metadata_assistant: data.data.metadata_assistant,
                content_type: data.data.content_type
            };
            displayAnalysisResults(analysisWithMetadata, currentContent, structuralBlocks);
        } else if (data.data.rewritten_text) {
            // Rewrite completed
            displayRewriteResults(data.data);
        }
    } else {
        // Show error
        showError('analysis-results', data.error || 'Process failed');
    }
}

// Create enhanced progress tracking display with PatternFly
function createProgressTracker(steps = []) {
    const defaultSteps = [
        { id: 'step-analysis', title: 'Content Analysis', description: 'Analyzing text structure and style' },
        { id: 'step-pass1', title: 'AI Processing', description: 'Generating improvements' },
        { id: 'step-pass2', title: 'Refinement', description: 'Polishing results' },
        { id: 'step-complete', title: 'Complete', description: 'Ready for review' }
    ];
    
    const stepsToUse = steps.length > 0 ? steps : defaultSteps;
    
    return `
        <div class="pf-v5-c-card app-card">
            <div class="pf-v5-c-card__header">
                <div class="pf-v5-c-card__header-main">
                    <h3 class="pf-v5-c-title pf-m-lg">
                        <i class="fas fa-tasks pf-v5-u-mr-sm" style="color: var(--app-primary-color);"></i>
                        Processing Progress
                    </h3>
                </div>
            </div>
            <div class="pf-v5-c-card__body">
                <div class="pf-v5-l-stack pf-m-gutter">
                    ${stepsToUse.map((step, index) => `
                        <div class="pf-v5-l-stack__item">
                            <div class="step-item pf-v5-c-card pf-m-plain pf-m-bordered" id="${step.id}">
                                <div class="pf-v5-c-card__body">
                                    <div class="pf-v5-l-flex pf-m-align-items-center">
                                        <div class="pf-v5-l-flex__item">
                                            <div class="step-icon" style="
                                                width: 40px;
                                                height: 40px;
                                                display: flex;
                                                align-items: center;
                                                justify-content: center;
                                                border-radius: var(--pf-v5-global--BorderRadius--lg);
                                                background: var(--pf-v5-global--BackgroundColor--200);
                                            ">
                                                <i class="fas fa-circle" style="color: var(--pf-v5-global--Color--200);"></i>
                                            </div>
                                        </div>
                                        <div class="pf-v5-l-flex__item pf-v5-u-ml-md">
                                            <h4 class="pf-v5-c-title pf-m-md pf-v5-u-mb-xs">${step.title}</h4>
                                            <p class="pf-v5-u-font-size-sm pf-v5-u-color-200 pf-v5-u-mb-0">${step.description}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
}

// Show real-time status updates with PatternFly alerts
function showStatusUpdate(message, type = 'info') {
    const statusContainer = document.getElementById('status-container');
    if (!statusContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `pf-v5-c-alert pf-m-${type} pf-m-inline fade-in`;
    alert.style.marginBottom = '1rem';
    
    alert.innerHTML = `
        <div class="pf-v5-c-alert__icon">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
        </div>
        <h4 class="pf-v5-c-alert__title">Processing Update</h4>
        <div class="pf-v5-c-alert__description">${message}</div>
    `;
    
    statusContainer.appendChild(alert);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.style.animation = 'fadeOut 0.3s ease-out forwards';
            setTimeout(() => alert.remove(), 300);
        }
    }, 3000);
}

// BLOCK-LEVEL REWRITING WEBSOCKET HANDLERS

/**
 * Handle block processing start event
 */
function handleBlockProcessingStart(data) {
    console.log('🚀 Block processing started:', data);
    
    const { block_id, block_type, applicable_stations } = data;
    
    // Initialize assembly line progress for this block
    initializeBlockAssemblyLineProgress(block_id, applicable_stations);
    
    // Update any global state if needed
    if (window.blockRewriteState) {
        window.blockRewriteState.currentlyProcessingBlock = block_id;
    }
}

/**
 * Handle station progress update
 */
function handleStationProgressUpdate(data) {
    console.log('🏭 Station progress update:', data);
    
    const { block_id, station, status, preview_text } = data;
    
    // Update specific station status in the assembly line UI
    updateStationStatus(block_id, station, status, preview_text);
    
    // Calculate overall progress based on completed stations
    const progressPercentage = calculateStationProgress(block_id, station, status);
    const statusText = generateProgressStatusText(station, status, preview_text);
    
    // Update overall assembly line progress
    updateBlockAssemblyLineProgress(block_id, progressPercentage, statusText);
    console.log(`🎯 Assembly line progress updated: ${block_id} = ${progressPercentage}% (${statusText})`);
}

/**
 * Calculate progress percentage based on station completion
 */
function calculateStationProgress(blockId, currentStation, status) {
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"] .assembly-line-stations`);
    if (!assemblyLineElement) return 0;
    
    const allStations = assemblyLineElement.querySelectorAll('[data-station]');
    const totalStations = allStations.length;
    
    if (totalStations === 0) return 100;
    
    let completedStations = 0;
    let currentStationProgress = 0;
    
    allStations.forEach(stationEl => {
        const stationName = stationEl.getAttribute('data-station');
        if (stationEl.classList.contains('station-complete')) {
            completedStations++;
        } else if (stationName === currentStation && status === 'processing') {
            currentStationProgress = 0.5; // 50% for current processing station
        }
    });
    
    // If current station just completed, count it
    if (status === 'complete' || status === 'completed') {
        completedStations++;
        currentStationProgress = 0;
    }
    
    const progress = ((completedStations + currentStationProgress) / totalStations) * 100;
    return Math.min(100, Math.max(0, progress));
}

/**
 * Generate status text for progress updates
 */
function generateProgressStatusText(station, status, previewText) {
    const stationNames = {
        'urgent': 'Critical/Legal Pass',
        'high': 'Structural Pass',
        'medium': 'Grammar Pass',
        'low': 'Style Pass'
    };
    
    const stationName = stationNames[station] || 'Processing Pass';
    
    if (status === 'processing') {
        return `🔄 ${stationName}: ${previewText || 'Processing...'}`;
    } else if (status === 'complete') {
        return `✅ ${stationName}: ${previewText || 'Complete'}`;
    }
    
    return `⏳ ${stationName}: Waiting...`;
}

/**
 * Handle block processing completion
 */
function handleBlockProcessingComplete(data) {
    console.log('✅ Block processing completed:', data);
    
    const { block_id, result } = data;
    
    // Update global state
    if (window.blockRewriteState) {
        window.blockRewriteState.processedBlocks.add(block_id);
        window.blockRewriteState.blockResults.set(block_id, result);
        window.blockRewriteState.currentlyProcessingBlock = null;
    }
    
    // Complete the assembly line UI
    completeBlockAssemblyLine(block_id);
    
    // The actual results display is handled by the main processing flow
    // This handler just manages the real-time progress UI
}

/**
 * Handle block processing error
 */
function handleBlockProcessingError(data) {
    console.error('❌ Block processing error:', data);
    
    const { block_id, error } = data;
    
    // Show error in assembly line UI
    showAssemblyLineError(block_id, error);
    
    // Update global state
    if (window.blockRewriteState) {
        window.blockRewriteState.currentlyProcessingBlock = null;
    }
}

/**
 * Initialize assembly line progress for a specific block
 */
function initializeBlockAssemblyLineProgress(blockId, stations) {
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"] .assembly-line-stations`);
    if (!assemblyLineElement) return;
    
    console.log(`🏗️ Creating ALL ${stations.length} stations upfront for block ${blockId}:`, stations);
    
    // Create ALL applicable stations immediately (not dynamically during processing)
    stations.forEach(station => {
        let stationElement = assemblyLineElement.querySelector(`[data-station="${station}"]`);
        
        // Create station if it doesn't exist
        if (!stationElement) {
            console.log(`🏗️ Creating missing station: ${station}`);
            
            // Get display name for this station
            const stationDisplayNames = {
                'urgent': 'Critical/Legal Pass',
                'high': 'Structural Pass',
                'medium': 'Grammar Pass',
                'low': 'Style Pass'
            };
            const displayName = stationDisplayNames[station] || 'Processing Pass';
            
            // Create HTML and insert into DOM (inline to avoid cross-file dependencies)
            const stationHtml = createStationElementHtml(station, displayName);
            assemblyLineElement.insertAdjacentHTML('beforeend', stationHtml);
            stationElement = assemblyLineElement.querySelector(`[data-station="${station}"]`);
        }
        
        // Reset station to waiting state
        if (stationElement) {
            stationElement.classList.remove('station-processing', 'station-complete', 'station-error');
            stationElement.classList.add('station-waiting');
            
            const statusIcon = stationElement.querySelector('.station-status-icon');
            if (statusIcon) {
                statusIcon.className = 'station-status-icon fas fa-clock';
            }
            
            const statusText = stationElement.querySelector('.station-status-text');
            if (statusText) {
                statusText.textContent = 'Waiting...';
            }
        }
    });
    
    console.log(`✅ ALL ${stations.length} stations displayed upfront and ready for processing`);
}

/**
 * Create HTML for a station element (inline version to avoid cross-file dependencies)
 */
function createStationElementHtml(station, displayName) {
    // Station color mapping
    const getStationColor = (station) => {
        const colorMap = { 'urgent': 'red', 'high': 'orange', 'medium': 'gold', 'low': 'blue' };
        return colorMap[station] || 'grey';
    };
    
    // Station priority mapping  
    const getStationPriority = (station) => {
        const priorityMap = { 'urgent': 'Critical', 'high': 'High', 'medium': 'Medium', 'low': 'Low' };
        return priorityMap[station] || 'Normal';
    };
    
    return `
        <div class="pf-v5-l-stack__item">
            <div class="dynamic-station station-waiting" data-station="${station}">
                <div class="pf-v5-l-flex pf-m-align-items-center">
                    <div class="pf-v5-l-flex__item">
                        <i class="station-status-icon fas fa-clock pf-v5-u-mr-sm"></i>
                    </div>
                    <div class="pf-v5-l-flex__item pf-m-flex-1">
                        <span class="pf-v5-u-font-weight-bold">${displayName}</span>
                        <div class="station-status-text pf-v5-u-font-size-sm pf-v5-u-color-400">
                            Waiting...
                        </div>
                    </div>
                    <div class="pf-v5-l-flex__item">
                        <span class="station-priority-badge pf-v5-c-label pf-m-outline pf-m-${getStationColor(station)}">
                            <span class="pf-v5-c-label__content">${getStationPriority(station)}</span>
                        </span>
                    </div>
                </div>
                <div class="station-preview pf-v5-u-mt-xs pf-v5-u-font-size-sm pf-v5-u-color-300" style="display: none;">
                    <!-- Preview text will be inserted here during processing -->
                </div>
            </div>
        </div>
    `;
}

/**
 * Update status of a specific station
 */
function updateStationStatus(blockId, station, status, previewText = null) {
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"] .assembly-line-stations`);
    if (!assemblyLineElement) return;
    
    let stationElement = assemblyLineElement.querySelector(`[data-station="${station}"]`);
    
    // Since all stations are now shown upfront, this should not happen
    if (!stationElement) {
        console.warn(`⚠️ Station element not found for: ${station} (this should not happen with upfront station display)`);
        return;
    }
    
    // Update station classes
    stationElement.classList.remove('station-waiting', 'station-processing', 'station-complete', 'station-error');
    
    const statusIcon = stationElement.querySelector('.station-status-icon');
    const statusText = stationElement.querySelector('.station-status-text');
    
    switch (status) {
        case 'processing':
            stationElement.classList.add('station-processing');
            if (statusIcon) statusIcon.className = 'station-status-icon fas fa-spinner fa-spin';
            if (statusText) statusText.textContent = 'Processing...';
            break;
        case 'complete':
        case 'completed':  // Handle both status values for consistency
            stationElement.classList.add('station-complete');
            if (statusIcon) statusIcon.className = 'station-status-icon fas fa-check-circle';
            if (statusText) statusText.textContent = 'Complete';
            break;
        case 'error':
            stationElement.classList.add('station-error');
            if (statusIcon) statusIcon.className = 'station-status-icon fas fa-exclamation-triangle';
            if (statusText) statusText.textContent = 'Error';
            break;
        default:
            stationElement.classList.add('station-waiting');
            if (statusIcon) statusIcon.className = 'station-status-icon fas fa-clock';
            if (statusText) statusText.textContent = 'Waiting';
    }
    
    // Update preview text if provided
    if (previewText) {
        const previewElement = stationElement.querySelector('.station-preview');
        if (previewElement) {
            previewElement.textContent = previewText;
        }
    }
}

/**
 * Update overall assembly line progress
 */
function updateBlockAssemblyLineProgress(blockId, progressPercent, statusText) {
    console.log(`🎯 Updating assembly line progress: ${blockId} → ${progressPercent}% → "${statusText}"`);
    
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"].block-assembly-line`);
    if (!assemblyLineElement) {
        console.warn(`⚠️  Assembly line element not found for block: ${blockId}`);
        return;
    }
    
    // Ensure progress is valid
    const validProgress = Math.max(0, Math.min(100, progressPercent || 0));
    
    // Update progress bar
    const progressBar = assemblyLineElement.querySelector('.pf-v5-c-progress__bar');
    if (progressBar) {
        progressBar.style.width = `${validProgress}%`;
        console.log(`📊 Progress bar updated to ${validProgress}%`);
    } else {
        console.warn(`⚠️  Progress bar element not found in assembly line for block: ${blockId}`);
    }
    
    // Update progress text
    const progressText = assemblyLineElement.querySelector('.assembly-line-status');
    if (progressText) {
        progressText.textContent = statusText || 'Processing...';
    }
    
    // Update progress percentage text
    const progressPercentElement = assemblyLineElement.querySelector('.progress-percent');
    if (progressPercentElement) {
        progressPercentElement.textContent = `${Math.round(validProgress)}%`;
        console.log(`🔢 Progress percentage updated to ${Math.round(validProgress)}%`);
    } else {
        console.warn(`⚠️  Progress percent element not found in assembly line for block: ${blockId}`);
    }
}

/**
 * Complete assembly line processing
 */
function completeBlockAssemblyLine(blockId) {
    updateBlockAssemblyLineProgress(blockId, 100, 'Processing complete');
    
    // Add completion animation
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"].block-assembly-line`);
    if (assemblyLineElement) {
        assemblyLineElement.classList.add('assembly-line-complete');
        
        // Auto-hide assembly line after showing completion
        setTimeout(() => {
            if (assemblyLineElement.parentNode) {
                assemblyLineElement.style.opacity = '0.7';
            }
        }, 2000);
    }
}

/**
 * Show error in assembly line
 */
function showAssemblyLineError(blockId, errorMessage) {
    const assemblyLineElement = document.querySelector(`[data-block-id="${blockId}"].block-assembly-line`);
    if (!assemblyLineElement) return;
    
    // Update status to error
    const statusElement = assemblyLineElement.querySelector('.assembly-line-status');
    if (statusElement) {
        statusElement.textContent = `Error: ${errorMessage}`;
        statusElement.classList.add('error-text');
    }
    
    // Add error styling
    assemblyLineElement.classList.add('assembly-line-error');
}

/**
 * Join session room dynamically
 */
function joinSessionRoom(sessionId) {
    if (socket && socket.connected && sessionId) {
        console.log(`🔍 DEBUG: Dynamically joining session room: ${sessionId}`);
        socket.emit('join_session', { session_id: sessionId });
        return true;
    } else {
        console.warn(`❌ Cannot join session room: socket=${!!socket}, connected=${socket?.connected}, sessionId=${sessionId}`);
        return false;
    }
}

// Export functions for global use
if (typeof window !== 'undefined') {
    window.joinSessionRoom = joinSessionRoom;
} 