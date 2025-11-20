document.addEventListener('DOMContentLoaded', function() {
  const algorithmSelect = document.getElementById('algorithm');
  
  const ajusteSection = document.getElementById('ajusteSection');
  const clusteringSection = document.getElementById('clusteringSection');
  const chiSection = document.getElementById('chiSection');
  const arbolSection = document.getElementById('arbolSection');
  
  algorithmSelect.addEventListener('change', function() {
    const algorithm = this.value;
    
    hideAllSections();
    
    switch(algorithm) {
      case 'kmedias':
        ajusteSection.classList.remove('hidden');
        clusteringSection.classList.remove('hidden');
        break;
      
      case 'kmodas':
        clusteringSection.classList.remove('hidden');
        break;
      
      case 'chiagrup':
        chiSection.classList.remove('hidden');
        break;
      
      case 'arbol':
        ajusteSection.classList.remove('hidden');
        arbolSection.classList.remove('hidden');
        break;
      
      case 'soloajuste':
        ajusteSection.classList.remove('hidden');
        break;
      
      default:
        break;
    }
  });

  function hideAllSections() {
    ajusteSection.classList.add('hidden');
    clusteringSection.classList.add('hidden');
    chiSection.classList.add('hidden');
    arbolSection.classList.add('hidden');
  }
});
