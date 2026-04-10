// Initialize Particles.js (using tsParticles bundle)
tsParticles.load("particles-js", {
    background: {
        color: {
            value: "transparent",
        },
    },
    fpsLimit: 60,
    interactivity: {
        events: {
            onHover: {
                enable: true,
                mode: "grab",
            },
        },
        modes: {
            grab: {
                distance: 140,
                links: {
                    opacity: 0.8
                }
            }
        }
    },
    particles: {
        color: {
            value: "#a855f7",
        },
        links: {
            color: "#3b82f6",
            distance: 150,
            enable: true,
            opacity: 0.3,
            width: 1.5,
        },
        move: {
            direction: "none",
            enable: true,
            outModes: {
                default: "bounce",
            },
            random: true,
            speed: 1.2,
            straight: false,
        },
        number: {
            density: {
                enable: true,
                area: 800,
            },
            value: 60,
        },
        opacity: {
            value: 0.6,
        },
        shape: {
            type: "circle",
        },
        size: {
            value: { min: 1, max: 3 },
        },
    },
    detectRetina: true,
});

// Initialize Three.js
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Create an Icosahedron to represent the "Brain" or "Core"
const geometry = new THREE.IcosahedronGeometry(2.5, 1);
const material = new THREE.MeshBasicMaterial({ 
    color: 0x3b82f6, 
    wireframe: true,
    transparent: true,
    opacity: 0.15
});
const sphere = new THREE.Mesh(geometry, material);
scene.add(sphere);

camera.position.z = 6;

// Track mouse for subtle rotation effect
let mouseX = 0;
let mouseY = 0;
document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
});

let isSpinningFast = false;

function animate() {
    requestAnimationFrame(animate);
    
    // Base rotation
    const rotateSpeed = isSpinningFast ? 0.05 : 0.002;
    sphere.rotation.x += rotateSpeed * 0.5;
    sphere.rotation.y += rotateSpeed;
    
    if (!isSpinningFast) {
        // Target rotation based on mouse (only when idle)
        sphere.rotation.x += (mouseY * 0.5 - sphere.rotation.x) * 0.02;
        sphere.rotation.y += (mouseX * 0.5 - sphere.rotation.y) * 0.02;
    }
    
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// App Logic (SSE Streaming)
const agentColors = {
    'planner_agent': '#3B82F6',
    'literature_agent': '#22C55E',
    'summarizer_agent': '#F97316',
    'conflict_resolver_agent': '#EF4444',
    'synthesizer_agent': '#A855F7',
    'evaluator_agent': '#14B8A6',
    'visualization_agent': '#EC4899',
    'credibility_scorer': '#6B7280'
};

document.getElementById('run-btn').addEventListener('click', async () => {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;
    const domain = document.getElementById('domain-mode').value;
    
    const logEl = document.getElementById('activity-log');
    const reportEl = document.getElementById('report-content');
    const btn = document.getElementById('run-btn');
    const metricsPanel = document.getElementById('metrics-panel');
    const metricsContent = document.getElementById('metrics-content');
    
    logEl.innerHTML = '';
    reportEl.innerHTML = '<i>Processing request...</i>';
    metricsPanel.style.display = 'none';
    
    btn.disabled = true;
    btn.innerText = 'Running...';
    isSpinningFast = true; // Speed up 3D sphere

    try {
        const response = await fetch(`/api/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, domain })
        });
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            let lines = buffer.split('\n\n');
            buffer = lines.pop() || ''; 
            
            for (let chunk of lines) {
                if (chunk.startsWith('data: ')) {
                    const dataStr = chunk.substring(6);
                    try {
                        const data = JSON.parse(dataStr);
                        
                        // Final result rendering
                        if (data.type === 'final_result') {
                            reportEl.innerHTML = marked.parse(data.report_markdown || '*No report generated.*');
                            
                            if (data.evaluator_output) {
                                metricsPanel.style.display = 'block';
                                metricsContent.innerHTML = `
                                    <div class="metric-card">
                                        <div class="metric-value">${(data.evaluator_output.composite_score * 100).toFixed(1)}/100</div>
                                        <div>Composite Score</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">${data.evaluator_output.coherence.toFixed(2)}</div>
                                        <div>Coherence</div>
                                    </div>
                                    <div class="metric-card">
                                        <div class="metric-value">${data.evaluator_output.factuality.toFixed(2)}</div>
                                        <div>Factuality</div>
                                    </div>
                                `;
                            }
                        } 
                        // Event log rendering
                        else if (data.agent_type) {
                            const bg = agentColors[data.agent_type] || '#6B7280';
                            const el = document.createElement('div');
                            el.className = 'event-row';
                            el.style.borderLeftColor = bg;
                            
                            const agentName = data.agent_type.replace('_agent','').replace('_', ' ').toUpperCase();
                            const icon = data.status === 'completed' ? '✅' : (data.status === 'failed' ? '❌' : '⏳');
                            
                            el.innerHTML = `
                                <span class="event-agent" style="background: ${bg}">${agentName}</span>
                                <span class="event-msg">${icon} ${data.message} <small style="display:block;color:rgba(255,255,255,0.4);margin-top:4px;">${data.timestamp.substring(11,19)}</small></span>
                            `;
                            
                            logEl.appendChild(el);
                            logEl.scrollTop = logEl.scrollHeight;
                        }
                    } catch(e) {
                        console.error('Parse err:', e, dataStr);
                    }
                }
            }
        }
    } catch (err) {
        reportEl.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
    } finally {
        isSpinningFast = false;
        btn.disabled = false;
        btn.innerText = 'Deploy Agents';
    }
});
