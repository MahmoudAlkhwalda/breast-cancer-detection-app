// Admin Dashboard JavaScript
// Load training statistics
function loadTrainingStats() {
    fetch('/admin/training-stats')
        .then(response => response.json())
        .then(data => {
            alert('Training Stats:\n' + 
                  'Total Verified Data: ' + data.total_verified_data + '\n' +
                  'Form Data: ' + data.form_data + '\n' +
                  'Image Data: ' + data.image_data + '\n' +
                  'Unused Data: ' + data.unused_data);
        })
        .catch(error => console.error('Error:', error));
}

// Initialize prediction stats from server data
function initializePredictionStats() {
    if (window.predictionStats) {
        return window.predictionStats;
    } else {
        console.warn('predictionStats not found, using default values');
        return {
            form_predictions: 0,
            image_predictions: 0,
            benign_predictions: 0,
            malignant_predictions: 0,
            normal_predictions: 0
        };
    }
}

// Verify prediction
function verifyPrediction(predictionId) {
    document.getElementById('verifyForm').action = '/admin/verify-prediction/' + predictionId;
    var modal = new bootstrap.Modal(document.getElementById('verifyModal'));
    modal.show();
}

// Initialize charts with data from server
function initializeCharts(chartData) {
    // Prediction Types Chart
    const ctx1 = document.getElementById('predictionTypesChart').getContext('2d');
    new Chart(ctx1, {
        type: 'doughnut',
        data: {
            labels: ['Form Predictions', 'Image Predictions'],
            datasets: [{
                data: [chartData.formPredictions, chartData.imagePredictions],
                backgroundColor: ['#007bff', '#17a2b8'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Diagnosis Results Chart
    const ctx2 = document.getElementById('diagnosisChart').getContext('2d');
    new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: ['Benign', 'Malignant', 'Normal'],
            datasets: [{
                label: 'Count',
                data: [chartData.benignPredictions, chartData.malignantPredictions, chartData.normalPredictions],
                backgroundColor: ['#28a745', '#dc3545', '#6c757d'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get prediction stats
    const stats = initializePredictionStats();
    
    // Chart data from server
    const chartData = {
        formPredictions: stats.form_predictions || 0,
        imagePredictions: stats.image_predictions || 0,
        benignPredictions: stats.benign_predictions || 0,
        malignantPredictions: stats.malignant_predictions || 0,
        normalPredictions: stats.normal_predictions || 0
    };
    
    // Initialize charts
    initializeCharts(chartData);
    
    // Add event listeners for verify buttons
    document.querySelectorAll('.admin-verify-btn').forEach(button => {
        button.addEventListener('click', function() {
            const predictionId = this.getAttribute('data-prediction-id');
            verifyPrediction(predictionId);
        });
    });
    
    // Add confidence bar colors
    document.querySelectorAll('.confidence-bar').forEach(bar => {
        const confidence = parseFloat(bar.getAttribute('data-confidence'));
        if (confidence > 70) {
            bar.classList.add('bg-success');
        } else if (confidence > 50) {
            bar.classList.add('bg-warning');
        } else {
            bar.classList.add('bg-danger');
        }
    });
});
