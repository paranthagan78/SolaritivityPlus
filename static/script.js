// Global variables
let uploadedImagePath = null;
let currentClassification = null;
let uploadedFile = null;

// DOM elements
const uploadBtn = document.getElementById('uploadBtn');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const classifyBtn = document.getElementById('classifyBtn');
const detectBtn = document.getElementById('detectBtn');
const classificationSection = document.getElementById('classificationSection');
const detectionSection = document.getElementById('detectionSection');
const explainabilitySection = document.getElementById('explainabilitySection');
const explainBtn = document.getElementById('explainBtn');

// Panel configuration elements
const citySelect = document.getElementById('citySelect');
const panelPower = document.getElementById('panelPower');
const ambientTemp = document.getElementById('ambientTemp');
const irradiance = document.getElementById('irradiance');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing...');
    
    // Event listeners
    uploadBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('Upload button clicked');
        imageInput.click();
    });
    
    imageInput.addEventListener('change', handleImageUpload);
    classifyBtn.addEventListener('click', runClassification);
    detectBtn.addEventListener('click', runDetection);
    explainBtn.addEventListener('click', generateExplanation);

    // Opacity slider for explanation
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityValue = document.getElementById('opacityValue');
    
    if (opacitySlider) {
        opacitySlider.addEventListener('input', function() {
            const value = this.value;
            opacityValue.textContent = value + '%';
            const explainImage = document.getElementById('explainImage');
            if (explainImage) {
                explainImage.style.opacity = value / 100;
            }
        });
    }
    
    console.log('Initialization complete');
});

// Handle image upload
function handleImageUpload(event) {
    console.log('Image upload triggered');
    const file = event.target.files[0];
    if (!file) {
        console.log('No file selected');
        return;
    }

    console.log('File selected:', file.name, file.type, file.size);

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size should be less than 10MB');
        return;
    }

    // Store file for later use
    uploadedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        console.log('Image loaded for preview');
        imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        classifyBtn.disabled = false;
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    // Reset sections
    classificationSection.style.display = 'none';
    detectionSection.style.display = 'none';
    explainabilitySection.style.display = 'none';
}

// Run classification only
async function runClassification() {
    if (!uploadedFile) {
        alert('Please select an image first');
        return;
    }

    // Disable button and show loading
    classifyBtn.disabled = true;
    classifyBtn.textContent = '🔄 Classifying...';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', uploadedFile);

        console.log('Sending classification request...');

        // Send request
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Classification failed');
        }

        const data = await response.json();
        console.log('Classification response:', data);

        if (!data.success) {
            throw new Error(data.error || 'Classification failed');
        }

        // Store classification data
        currentClassification = data.prediction;
        uploadedImagePath = data.image_path;

        // Display classification results
        displayClassificationResults(data);

        // Show classification section
        classificationSection.style.display = 'block';

        // Scroll to results
        classificationSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Classification error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        classifyBtn.disabled = false;
        classifyBtn.textContent = '🎯 Classify Defect';
    }
}

// Run detection with carbon emission analysis
async function runDetection() {
    if (!uploadedFile) {
        alert('Please select an image first');
        return;
    }

    // Disable button and show loading
    detectBtn.disabled = true;
    detectBtn.textContent = '🔄 Detecting & Analyzing...';

    try {
        // Create form data with panel configuration
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('city', citySelect.value);
        formData.append('panel_power', panelPower.value);
        formData.append('ambient_temp', ambientTemp.value);
        formData.append('irradiance', irradiance.value);

        console.log('Sending detection request...');
        console.log('Configuration:', {
            city: citySelect.value,
            panel_power: panelPower.value,
            ambient_temp: ambientTemp.value,
            irradiance: irradiance.value
        });

        // Send request
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Detection failed');
        }

        const data = await response.json();
        console.log('Detection response:', data);

        if (!data.success) {
            throw new Error(data.error || 'Detection failed');
        }

        // Display detection results
        displayDetectionResults(data);

        // Display carbon emission results if available
        if (data.carbon_emission) {
            displayCarbonEmission(data.carbon_emission);
        }

        // Show detection section
        detectionSection.style.display = 'block';

        // Scroll to results
        detectionSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Detection error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = '🔍 Detect & Analyze Carbon Impact';
    }
}

// Display classification results
function displayClassificationResults(data) {
    try {
        const prediction = data.prediction;
        
        // Main prediction
        const defectLabel = document.getElementById('defectLabel');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidenceFill = document.getElementById('confidenceFill');

        const label = prediction.label || 'Unknown';
        const confidence = (prediction.confidence * 100).toFixed(2);

        defectLabel.textContent = formatLabel(label);
        confidenceValue.textContent = confidence + '%';
        confidenceFill.style.width = confidence + '%';

        // Set color based on confidence
        if (confidence >= 80) {
            confidenceFill.style.backgroundColor = '#4caf50';
        } else if (confidence >= 60) {
            confidenceFill.style.backgroundColor = '#ff9800';
        } else {
            confidenceFill.style.backgroundColor = '#f44336';
        }

        // All predictions
        const allPredictionsDiv = document.getElementById('allPredictions');
        allPredictionsDiv.innerHTML = '';

        if (data.all_predictions && Array.isArray(data.all_predictions)) {
            data.all_predictions.forEach(pred => {
                const predDiv = document.createElement('div');
                predDiv.className = 'prediction-item';
                const predConfidence = (pred.confidence * 100).toFixed(2);
                predDiv.innerHTML = `
                    <span class="pred-label">${formatLabel(pred.label)}</span>
                    <span class="pred-confidence">${predConfidence}%</span>
                    <div class="pred-bar">
                        <div class="pred-bar-fill" style="width: ${predConfidence}%"></div>
                    </div>
                `;
                allPredictionsDiv.appendChild(predDiv);
            });
        }

    } catch (error) {
        console.error('Error displaying classification results:', error);
        alert('Error displaying classification results. Please try again.');
    }
}

// Display detection results
function displayDetectionResults(data) {
    try {
        // Update detection count
        const defectCount = document.getElementById('defectCount');
        defectCount.textContent = data.count || 0;

        // Display images
        const originalImage = document.getElementById('originalImage');
        const detectionImage = document.getElementById('detectionImage');
        
        if (data.original_image) {
            originalImage.src = data.original_image;
        }
        
        if (data.result_image) {
            detectionImage.src = data.result_image;
        }

        // Display detection details
        const detectionsList = document.getElementById('detectionsList');
        detectionsList.innerHTML = '';

        if (data.detections && data.detections.length > 0) {
            data.detections.forEach((det, index) => {
                const detDiv = document.createElement('div');
                detDiv.className = 'detection-item';
                detDiv.setAttribute('data-class', det.class);
                
                const confidence = (det.confidence * 100).toFixed(2);
                const bboxStr = `[${det.bbox.join(', ')}]`;
                
                detDiv.innerHTML = `
                    <div class="detection-index">${index + 1}</div>
                    <div class="detection-info-text">
                        <div class="detection-class-name">${formatLabel(det.class)}</div>
                        <div class="detection-bbox">Location: ${bboxStr}</div>
                    </div>
                    <div class="detection-confidence-value">${confidence}%</div>
                `;
                
                detectionsList.appendChild(detDiv);
            });
        } else {
            detectionsList.innerHTML = `
                <div class="no-detections">
                    <p>✅ No defects detected in this image.</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Error displaying detection results:', error);
        alert('Error displaying detection results. Please try again.');
    }
}

// Display carbon emission results
function displayCarbonEmission(carbonData) {
    try {
        const carbonSection = document.getElementById('carbonSection');
        carbonSection.style.display = 'block';

        // Main carbon value
        const carbonValue = document.getElementById('carbonValue');
        carbonValue.textContent = carbonData.co2_emission_kg_per_year;

        // Metrics
        document.getElementById('degradationValue').textContent = 
            carbonData.total_degradation + '%';
        document.getElementById('dominantDefectValue').textContent = 
            formatLabel(carbonData.dominant_defect);
        document.getElementById('cityValue').textContent = 
            carbonData.city;
        document.getElementById('panelPowerValue').textContent = 
            carbonData.panel_power + 'W';

        // Interpretation
        const interpretation = generateCarbonInterpretation(carbonData);
        document.getElementById('carbonInterpretation').textContent = interpretation;

        console.log('Carbon emission displayed:', carbonData);

    } catch (error) {
        console.error('Error displaying carbon emission:', error);
    }
}

// Generate carbon emission interpretation
function generateCarbonInterpretation(carbonData) {
    const co2 = carbonData.co2_emission_kg_per_year;
    const degradation = carbonData.total_degradation;
    const numDefects = carbonData.num_defects;
    
    let interpretation = '';
    
    if (numDefects === 0) {
        interpretation = '✅ No defects detected! This panel is operating at optimal efficiency with minimal carbon emission impact.';
    } else if (degradation < 5) {
        interpretation = `⚠️ Minor defects detected (${numDefects} defect${numDefects > 1 ? 's' : ''}). ` +
            `The panel shows ${degradation}% degradation, resulting in an estimated ${co2} kg CO₂/year emission. ` +
            `Consider monitoring but immediate action may not be required.`;
    } else if (degradation < 15) {
        interpretation = `⚠️ Moderate defects detected (${numDefects} defect${numDefects > 1 ? 's' : ''}). ` +
            `The panel shows ${degradation}% degradation, resulting in an estimated ${co2} kg CO₂/year emission. ` +
            `Maintenance is recommended to restore efficiency.`;
    } else {
        interpretation = `🚨 Significant defects detected (${numDefects} defect${numDefects > 1 ? 's' : ''}). ` +
            `The panel shows ${degradation}% degradation, resulting in an estimated ${co2} kg CO₂/year emission. ` +
            `Immediate maintenance or replacement is strongly recommended to minimize environmental impact.`;
    }
    
    return interpretation;
}

// Format label for display
function formatLabel(label) {
    if (!label) return 'Unknown';
    return label.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

// Generate LIME explanation
async function generateExplanation() {
    if (!uploadedImagePath) {
        alert('Please run classification first');
        return;
    }

    const explainLoading = document.getElementById('explainLoading');
    const explainContent = document.getElementById('explainContent');

    // Show explainability section and loading
    explainabilitySection.style.display = 'block';
    explainBtn.disabled = true;
    explainLoading.style.display = 'block';
    explainContent.style.display = 'none';

    try {
        console.log('Sending explanation request...');
        const response = await fetch('/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: uploadedImagePath
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate explanation');
        }

        const data = await response.json();
        console.log('Explanation response:', data);

        if (!data.success) {
            throw new Error(data.error || 'Failed to generate explanation');
        }

        // Display explanation
        const explainImage = document.getElementById('explainImage');
        explainImage.src = data.explanation_image;
        explainImage.onload = () => {
            explainContent.style.display = 'block';
            explainabilitySection.scrollIntoView({ behavior: 'smooth' });
        };

        // Display features
        const featuresList = document.getElementById('featuresList');
        featuresList.innerHTML = '';

        if (data.features && Array.isArray(data.features)) {
            data.features.forEach(feature => {
                const featureDiv = document.createElement('div');
                featureDiv.className = 'feature-item';
                const contributionClass = feature.contribution === 'Positive' ? 'positive' : 'negative';
                featureDiv.innerHTML = `
                    <div class="feature-name">${feature.feature}</div>
                    <div class="feature-contribution ${contributionClass}">
                        ${feature.contribution}
                    </div>
                    <div class="feature-importance">
                        Importance: ${feature.importance.toFixed(4)}
                    </div>
                `;
                featuresList.appendChild(featureDiv);
            });
        }

    } catch (error) {
        console.error('Explanation error:', error);
        alert(`Error generating explanation: ${error.message}`);
    } finally {
        explainLoading.style.display = 'none';
        explainBtn.disabled = false;
    }
}
