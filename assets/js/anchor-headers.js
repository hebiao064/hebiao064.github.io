document.addEventListener('DOMContentLoaded', function() {
  // Select all h1 and h2 elements within articles
  const headers = document.querySelectorAll('article h1, article h2');
  
  headers.forEach(function(header) {
    // Skip headers that already have anchors
    if (header.querySelector('a')) return;
    
    // Create an id from the header text
    const id = header.textContent.trim()
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')  // Remove special characters
      .replace(/\s+/g, '-');     // Replace spaces with hyphens
    
    // Set the id on the header
    header.id = id;
    
    // Create a wrapper for the header content
    const wrapper = document.createElement('a');
    wrapper.href = '#' + id;
    
    // Clone all children of the header
    while (header.firstChild) {
      wrapper.appendChild(header.firstChild);
    }
    
    // Append the wrapper back to the header
    header.appendChild(wrapper);
  });
}); 