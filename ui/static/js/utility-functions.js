// Enhanced Utility functions for PatternFly UI

// Format rule type for display - handles special cases like word usage rules
function formatRuleType(ruleType) {
    if (!ruleType) return 'Style Issue';
    
    // Group all word usage rules under "Word Usage"
    if (ruleType.startsWith('word_usage_')) {
        return 'Word Usage';
    }
    
    // Group all numbers rules under better names
    if (ruleType.startsWith('numbers_')) {
        const numberTypeMap = {
            'numbers_currency': 'Currency',
            'numbers_dates_times': 'Dates and Times', 
            'numbers_general': 'Numbers',
            'numbers_numerals_vs_words': 'Numbers vs Words',
            'numbers_units_of_measurement': 'Units of Measurement'
        };
        return numberTypeMap[ruleType] || 'Numbers';
    }
    
    // Group all references rules under better names
    if (ruleType.startsWith('references_')) {
        const referenceTypeMap = {
            'references_citations': 'Citations',
            'references_geographic_locations': 'Geographic Locations',
            'references_names_titles': 'Names and Titles',
            'references_product_names': 'Product Names',
            'references_product_versions': 'Product Versions'
        };
        return referenceTypeMap[ruleType] || 'References';
    }
    
    // Group all technical elements under better names
    if (ruleType.startsWith('technical_')) {
        const technicalTypeMap = {
            'technical_commands': 'Commands',
            'technical_files_directories': 'Files and Directories',
            'technical_keyboard_keys': 'Keyboard Keys',
            'technical_mouse_buttons': 'Mouse Buttons',
            'technical_programming_elements': 'Programming Elements',
            'technical_ui_elements': 'UI Elements',
            'technical_web_addresses': 'Web Addresses'
        };
        return technicalTypeMap[ruleType] || 'Technical Elements';
    }
    
    // Group all legal rules under better names
    if (ruleType.startsWith('legal_')) {
        const legalTypeMap = {
            'legal_claims': 'Claims',
            'legal_company_names': 'Company Names',
            'legal_personal_information': 'Personal Information'
        };
        return legalTypeMap[ruleType] || 'Legal Information';
    }
    
    // Group all audience rules under better names
    if (ruleType.startsWith('audience_')) {
        const audienceTypeMap = {
            'audience_tone': 'Tone',
            'audience_global': 'Global Audience',
            'audience_conversational': 'Conversational Style',
            'audience_llm_consumability': 'LLM Consumability'
        };
        return audienceTypeMap[ruleType] || 'Audience and Medium';
    }
    
    // Group all structure format rules under better names
    if (ruleType.startsWith('structure_format_')) {
        const structureTypeMap = {
            'structure_format_highlighting': 'Highlighting',
            'structure_format_glossaries': 'Glossaries'
        };
        return structureTypeMap[ruleType] || 'Structure and Format';
    }
    
    // Handle other specific cases
    const specificTypeMap = {
        'second_person': 'Second Person',
        'sentence_length': 'Sentence Length',
        'anthropomorphism': 'Anthropomorphism',
        'abbreviations': 'Abbreviations',
        'adverbs_only': 'Adverbs',
        'quotation_marks': 'Quotation Marks',
        'punctuation_and_symbols': 'Punctuation and Symbols',
        'exclamation_points': 'Exclamation Points'
    };
    
    if (specificTypeMap[ruleType]) {
        return specificTypeMap[ruleType];
    }
    
    // Default: replace underscores with spaces and title case
    return ruleType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Show loading state with LIVE DYNAMIC PROGRESS BAR (0-100%) - TRULY WORLD-CLASS VERSION
function showLoading(elementId, message = 'Processing...') {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.innerHTML = `
        <div class="pf-v5-c-card app-card" id="live-progress-container" style="width: 100%; max-width: none;">
            <div class="pf-v5-c-card__body" style="padding: 3rem 2rem;">
                
                <!-- Animated Spinner Icon -->
                <div style="text-align: center; margin-bottom: 2rem;">
                    <span class="pf-v5-c-spinner pf-m-xl" role="status" aria-label="Analyzing..." style="width: 64px; height: 64px;">
                        <span class="pf-v5-c-spinner__clipper"></span>
                        <span class="pf-v5-c-spinner__lead-ball"></span>
                        <span class="pf-v5-c-spinner__tail-ball"></span>
                    </span>
                </div>
                
                <!-- LIVE Status Message (DYNAMIC - changes based on stage) -->
                <div style="text-align: center; margin-bottom: 2.5rem;">
                    <h2 class="pf-v5-c-title pf-m-2xl pf-v5-u-mb-md" id="live-progress-status" 
                        style="color: var(--app-primary-color); font-size: 2rem; font-weight: 600; min-height: 3rem; display: flex; align-items: center; justify-content: center; transition: opacity 0.3s ease;">
                        Starting analysis...
                    </h2>
                    <p class="pf-v5-u-color-200 pf-v5-u-font-size-md" id="live-progress-detail" 
                       style="min-height: 2rem; font-size: 1.1rem; transition: opacity 0.3s ease;">
                        Preparing analysis pipeline
                    </p>
                </div>
                
                <!-- TRULY FULL WIDTH Progress Bar -->
                <div style="width: 100%; max-width: 100%; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span style="color: var(--pf-v5-global--Color--200); font-size: 0.9rem; font-weight: 500;">
                            Analysis Progress
                        </span>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span id="live-elapsed-time" style="font-size: 0.9rem; color: var(--pf-v5-global--Color--200); font-family: monospace;">
                                0:00
                            </span>
                            <span id="live-progress-percentage" style="font-size: 1.5rem; font-weight: 700; color: var(--app-primary-color); min-width: 60px; text-align: right;">
                                0%
                            </span>
                        </div>
                    </div>
                    <div role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="live-progress-bar-element" 
                         style="width: 100%; height: 16px; background-color: rgba(0, 0, 0, 0.06); border-radius: 8px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                        <div id="live-progress-indicator" 
                             style="width: 0%; height: 100%; background: linear-gradient(90deg, #0088ff, #0066cc); transition: width 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94), background 0.5s ease; box-shadow: 0 0 10px rgba(0, 136, 255, 0.3); will-change: width;">
                        </div>
                    </div>
                </div>
                
                <!-- LIVE Stage Badge - Shows ONLY current stage -->
                <div id="live-stage-display" style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(0, 102, 204, 0.08), rgba(0, 102, 204, 0.04)); border-radius: 10px; margin-top: 2rem; border: 1px solid rgba(0, 102, 204, 0.15); min-height: 90px; display: flex; align-items: center; justify-content: center;">
                    <div style="display: inline-flex; align-items: center; gap: 1rem;">
                        <i class="fas fa-spinner fa-pulse" id="live-stage-icon" style="color: var(--app-primary-color); font-size: 1.3rem; transition: all 0.3s ease;"></i>
                        <span id="live-stage-text" style="font-size: 1.3rem; color: var(--app-primary-color); font-weight: 700; transition: transform 0.2s ease, color 0.3s ease;">
                            Initializing...
                        </span>
                    </div>
                </div>
                
            </div>
        </div>
    `;
    
    // CRITICAL: Use setTimeout to ensure DOM is fully rendered before WebSocket updates arrive
    setTimeout(() => {
        // Start elapsed time counter
        window.analysisStartTime = Date.now();
        window.elapsedTimeInterval = setInterval(updateElapsedTime, 500); // Update every 500ms
        
        // Mark that progress UI is ready
        window.progressUIReady = true;
        console.log('✅ Progress UI fully rendered and ready for WebSocket updates');
        
        // Start fallback progress animation if WebSocket is slow
        window.progressAnimationInterval = setTimeout(() => {
            startFallbackProgressAnimation();
        }, 1500); // Start after 1.5 seconds if no WebSocket updates
    }, 50); // Small delay to ensure DOM rendering
}

// Update elapsed time display
function updateElapsedTime() {
    if (!window.analysisStartTime) return;
    
    const elapsed = Math.floor((Date.now() - window.analysisStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    
    const elapsedElement = document.getElementById('live-elapsed-time');
    if (elapsedElement) {
        elapsedElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
}

// Fallback progress animation if WebSocket is slow - SMOOTH and REALISTIC
function startFallbackProgressAnimation() {
    let currentProgress = parseInt(document.getElementById('live-progress-bar-element')?.getAttribute('aria-valuenow')) || 0;
    
    window.fallbackProgressInterval = setInterval(() => {
        const indicator = document.getElementById('live-progress-indicator');
        const percentage = document.getElementById('live-progress-percentage');
        const progressBar = document.getElementById('live-progress-bar-element');
        
        if (!indicator || !percentage) {
            clearInterval(window.fallbackProgressInterval);
            return;
        }
        
        // Get current value to avoid going backwards
        const currentValue = parseInt(progressBar?.getAttribute('aria-valuenow')) || 0;
        
        // Only increment if fallback is still active (not replaced by real progress)
        if (currentValue >= currentProgress) {
            currentProgress = currentValue;
        }
        
        // Simulate realistic progress (slower as it approaches 100%)
        if (currentProgress < 30) {
            currentProgress += 1.5;
        } else if (currentProgress < 60) {
            currentProgress += 1;
        } else if (currentProgress < 80) {
            currentProgress += 0.5;
        } else {
            currentProgress += 0.25;
        }
        
        currentProgress = Math.min(currentProgress, 90); // Cap at 90% until real completion
        
        // Only update if we're moving forward
        if (currentProgress > currentValue) {
            indicator.style.width = `${currentProgress}%`;
            percentage.textContent = `${Math.round(currentProgress)}%`;
            progressBar.setAttribute('aria-valuenow', Math.round(currentProgress));
        }
        
    }, 600); // Update every 600ms for smoother animation
}

// Clear all progress timers
function clearProgressTimers() {
    if (window.elapsedTimeInterval) {
        clearInterval(window.elapsedTimeInterval);
        window.elapsedTimeInterval = null;
    }
    if (window.progressAnimationInterval) {
        clearTimeout(window.progressAnimationInterval);
        window.progressAnimationInterval = null;
    }
    if (window.fallbackProgressInterval) {
        clearInterval(window.fallbackProgressInterval);
        window.fallbackProgressInterval = null;
    }
}

// Show error message using enhanced PatternFly Alert
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.innerHTML = `
        <div class="pf-v5-c-alert pf-m-danger pf-m-inline fade-in" role="alert">
            <div class="pf-v5-c-alert__icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <h4 class="pf-v5-c-alert__title">Analysis Error</h4>
            <div class="pf-v5-c-alert__description">
                <p>${message}</p>
                <div class="pf-v5-u-mt-sm">
                    <button class="pf-v5-c-button pf-m-link pf-m-inline" type="button" onclick="location.reload()">
                        <i class="fas fa-redo pf-v5-u-mr-xs"></i>
                        Try Again
                    </button>
                </div>
            </div>
            <div class="pf-v5-c-alert__action">
                <button class="pf-v5-c-button pf-m-plain" type="button" onclick="this.closest('.pf-v5-c-alert').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
}

// Show success message using enhanced PatternFly Alert
function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.innerHTML = `
        <div class="pf-v5-c-alert pf-m-success pf-m-inline fade-in" role="alert">
            <div class="pf-v5-c-alert__icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h4 class="pf-v5-c-alert__title">${message}</h4>
            <div class="pf-v5-c-alert__action">
                <button class="pf-v5-c-button pf-m-plain" type="button" onclick="this.closest('.pf-v5-c-alert').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
    `;
}

// Basic highlight errors function (enhanced version is in error-display.js)
function highlightErrors(text, errors) {
    if (!errors || errors.length === 0) return escapeHtml(text);
    
    let highlightedText = escapeHtml(text);
    
    // Simple highlighting - just mark error text
    errors.forEach(error => {
        if (error.text_segment) {
            const segment = escapeHtml(error.text_segment);
            const highlightedSegment = `<mark style="background-color: rgba(201, 25, 11, 0.1); border-bottom: 2px solid var(--app-danger-color);">${segment}</mark>`;
            highlightedText = highlightedText.replace(segment, highlightedSegment);
        }
    });
    
    return highlightedText;
}

// Basic error card function (enhanced version is in error-display.js)
function createErrorCard(error, index = 0) {
    const errorType = error.error_type || 'STYLE';
    const message = error.message || 'Style issue detected';
    
    return `
        <div class="pf-v5-c-alert pf-m-warning pf-m-inline pf-v5-u-mb-sm">
            <div class="pf-v5-c-alert__icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="pf-v5-c-alert__title">${formatRuleType(errorType)}</div>
            <div class="pf-v5-c-alert__description">${message}</div>
        </div>
    `;
}
