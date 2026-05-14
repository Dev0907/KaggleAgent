document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const compUrlInput = document.getElementById('competition-url');
    const agentStatus = document.getElementById('agent-status');
    
    // Graph Nodes mapped to the 5-Agent Architecture
    const nodes = [
        { id: 'node-explanation', name: 'Overview', elem: document.getElementById('node-explanation') },
        { id: 'node-data', name: 'Data', elem: document.getElementById('node-data') },
        { id: 'node-approaches', name: 'Past Approaches', elem: document.getElementById('node-approaches') },
        { id: 'node-winners', name: 'Winner Insights', elem: document.getElementById('node-winners') },
        { id: 'node-discussion', name: 'Forum Intel', elem: document.getElementById('node-discussion') }
    ];
    
    const nodeMapping = {
        'explanation': 0,
        'data': 1,
        'approaches': 2,
        'winners': 3,
        'discussion': 4
    };
    
    const edges = document.querySelectorAll('.edge');

    function resetGraph() {
        nodes.forEach(node => {
            node.elem.classList.remove('active', 'completed');
        });
        edges.forEach(edge => {
            edge.classList.remove('active', 'completed');
        });
        agentStatus.textContent = 'Waiting for task initialization...';
        
        // Hide all cards
        document.querySelectorAll('.info-card').forEach(card => {
            card.classList.remove('visible');
            card.querySelector('.card-content').innerHTML = '<i class="fa-solid fa-circle-notch fa-spin" style="color:var(--primary-color)"></i> Loading data...';
        });
    }

    function updateGraphNode(index, status) {
        if (index < 0 || index >= nodes.length) return;
        
        const node = nodes[index];
        
        if (status === 'active') {
            node.elem.classList.add('active');
            agentStatus.textContent = `Current Step: Analyzing ${node.name}...`;
            
            if (index > 0) {
                edges[index - 1].classList.add('active');
            }
        } else if (status === 'completed') {
            node.elem.classList.remove('active');
            node.elem.classList.add('completed');
            
            if (index > 0) {
                edges[index - 1].classList.remove('active');
                edges[index - 1].classList.add('completed');
            }
        }
    }

    // Use marked.js for proper markdown formatting
    function formatTextToHTML(text) {
        if(!text) return "";
        try {
            return marked.parse(text);
        } catch (e) {
            console.error("Markdown parsing error", e);
            return text;
        }
    }

    startBtn.addEventListener('click', async () => {
        const url = compUrlInput.value.trim();
        // Model is hardcoded in backend now, no need to send it from frontend

        if (!url) {
            alert('Please provide a Kaggle competition URL.');
            return;
        }

        startBtn.disabled = true;
        startBtn.classList.add('running');
        startBtn.innerHTML = '<span>Analyzing...</span> <i class="fa-solid fa-circle-notch fa-spin"></i>';
        
        resetGraph();
        agentStatus.textContent = `Initializing Analysis for ${url}`;

        try {
            const response = await fetch('http://localhost:8000/api/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: url
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let currentNodeIndex = -1;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim() !== '');
                
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === 'state_update') {
                            const nodeName = data.node;
                            
                            if (currentNodeIndex >= 0) {
                                updateGraphNode(currentNodeIndex, 'completed');
                            }
                            
                            currentNodeIndex = nodeMapping[nodeName];
                            
                            if (currentNodeIndex !== undefined) {
                                updateGraphNode(currentNodeIndex, 'active');
                            }
                        } 
                        else if (data.type === 'content_update') {
                            const cardId = `card-${data.section}`;
                            const contentId = `content-${data.section}`;
                            
                            const cardElem = document.getElementById(cardId);
                            const contentElem = document.getElementById(contentId);
                            
                            if (cardElem && contentElem) {
                                cardElem.classList.add('visible');
                                const htmlContent = data.content.includes('<') ? data.content : formatTextToHTML(data.content);
                                contentElem.innerHTML = htmlContent;
                            }
                        }
                        else if (data.type === 'result') {
                            if (currentNodeIndex >= 0) {
                                updateGraphNode(currentNodeIndex, 'completed');
                            }
                            agentStatus.textContent = 'Analysis Complete! All insights displayed below.';
                        }
                        else if (data.type === 'error') {
                            agentStatus.textContent = 'Agent encountered an error: ' + data.message;
                        }
                    } catch (e) {
                        // ignore non-json lines
                    }
                }
            }

        } catch (error) {
            agentStatus.textContent = 'Connection failed. Is the backend running?';
            console.error(error);
        } finally {
            startBtn.disabled = false;
            startBtn.classList.remove('running');
            startBtn.innerHTML = '<span>Analyze Competition</span> <i class="fa-solid fa-magnifying-glass"></i>';
        }
    });
});
