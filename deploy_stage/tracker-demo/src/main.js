/**
 * ShopDemo - Enhanced Main JavaScript
 * Product images, toasts, quantity, skeletons, animations
 */

// ================================
// PRODUCT DATA (with images & descriptions)
// ================================
const products = {
  1: {
    id: 1, name: 'Running Shoes', price: 99.99,
    image: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=300&fit=crop',
    description: 'Lightweight performance running shoes with responsive cushioning and breathable mesh upper. Perfect for daily training.',
    category: 'Footwear'
  },
  2: {
    id: 2, name: 'Wireless Headphones', price: 149.99,
    image: 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop',
    description: 'Premium noise-cancelling wireless headphones with 30-hour battery life and crystal-clear audio.',
    category: 'Electronics'
  },
  3: {
    id: 3, name: 'Smart Watch', price: 299.99,
    image: 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop',
    description: 'Advanced fitness tracking smartwatch with heart rate monitor, GPS, and 5-day battery life.',
    category: 'Electronics'
  },
  4: {
    id: 4, name: 'Laptop Bag', price: 79.99,
    image: 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=300&fit=crop',
    description: 'Water-resistant laptop bag with padded compartment, fits up to 15.6" laptops. Sleek and professional.',
    category: 'Accessories'
  },
  5: {
    id: 5, name: 'Bluetooth Speaker', price: 129.99,
    image: 'https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=300&fit=crop',
    description: 'Portable Bluetooth speaker with 360° sound, waterproof design, and 12-hour playtime.',
    category: 'Electronics'
  },
  6: {
    id: 6, name: 'Fitness Tracker', price: 89.99,
    image: 'https://images.unsplash.com/photo-1575311373937-040b8e1fd5b6?w=400&h=300&fit=crop',
    description: 'Slim fitness tracker with step counting, sleep monitoring, and smartphone notifications.',
    category: 'Electronics'
  }
};

// ================================
// CART STATE
// ================================
let cart = JSON.parse(localStorage.getItem('shopDemo_cart') || '[]');

// ================================
// TOAST NOTIFICATION SYSTEM
// ================================
function showToast(message, type = 'success') {
  // Remove existing toasts
  const existing = document.querySelector('.toast-notification');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast-notification toast-${type}`;

  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || icons.success}</span>
    <span class="toast-message">${message}</span>
  `;

  document.body.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => {
    toast.classList.add('toast-visible');
  });

  // Auto-remove
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ================================
// CART BADGE ANIMATION
// ================================
function animateCartBadge() {
  const badge = document.querySelector('.cart-link');
  if (badge) {
    badge.classList.add('cart-bounce');
    setTimeout(() => badge.classList.remove('cart-bounce'), 600);
  }
}

// ================================
// UPDATE CART COUNT
// ================================
function updateCartCount() {
  const countEl = document.getElementById('cart-count');
  if (countEl) {
    const totalItems = cart.reduce((sum, item) => sum + (item.qty || 1), 0);
    countEl.textContent = totalItems;
  }
}

// ================================
// ADD TO CART
// ================================
function addToCart(product) {
  // Check if product already in cart
  const existing = cart.find(item => item.id === product.id);
  if (existing) {
    existing.qty = (existing.qty || 1) + 1;
  } else {
    cart.push({ ...product, qty: 1 });
  }

  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();
  animateCartBadge();

  // Track event
  if (window.opEcomTracker) {
    window.opEcomTracker.trackEvent('add_to_cart', 'ecommerce', product.name, product.price);
  }

  showToast(`${product.name} added to cart!`, 'success');
}

// ================================
// CHANGE QUANTITY
// ================================
function changeQty(index, delta) {
  const item = cart[index];
  if (!item) return;

  item.qty = (item.qty || 1) + delta;

  if (item.qty <= 0) {
    cart.splice(index, 1);
    showToast(`${item.name} removed from cart`, 'info');
  }

  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();
  renderCart();
}

// ================================
// REMOVE FROM CART
// ================================
function removeFromCart(index) {
  const removed = cart.splice(index, 1)[0];
  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();

  if (window.opEcomTracker) {
    window.opEcomTracker.trackEvent('remove_from_cart', 'ecommerce', removed.name);
  }

  showToast(`${removed.name} removed from cart`, 'info');
  renderCart();
}

// ================================
// CLEAR CART
// ================================
function clearCart() {
  cart = [];
  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();
}

// ================================
// CART TOTAL
// ================================
function getCartTotal() {
  return cart.reduce((sum, item) => sum + (item.price * (item.qty || 1)), 0);
}

// ================================
// RENDER CART
// ================================
function renderCart() {
  const cartContainer = document.getElementById('cart-items');
  const cartSummary = document.getElementById('cart-summary');
  if (!cartContainer) return;

  if (cart.length === 0) {
    cartContainer.innerHTML = `
      <div class="empty-cart">
        <div class="icon">🛒</div>
        <h3>Your cart is empty</h3>
        <p class="text-muted">Add some products to get started</p>
        <a href="/products" class="btn btn-primary" style="margin-top:1rem;">Browse Products</a>
      </div>
    `;
    if (cartSummary) cartSummary.style.display = 'none';
    return;
  }

  cartContainer.innerHTML = cart.map((item, index) => `
    <div class="cart-item">
      <img src="${item.image}" alt="${item.name}" class="cart-item-image">
      <div class="cart-item-details">
        <div class="cart-item-name">${item.name}</div>
        <div class="cart-item-price">$${(item.price * (item.qty || 1)).toFixed(2)}</div>
      </div>
      <div class="qty-controls">
        <button class="qty-btn" onclick="changeQty(${index}, -1)">−</button>
        <span class="qty-value">${item.qty || 1}</span>
        <button class="qty-btn" onclick="changeQty(${index}, 1)">+</button>
      </div>
      <button class="btn-remove" onclick="removeFromCart(${index})" title="Remove">✕</button>
    </div>
  `).join('');

  if (cartSummary) {
    cartSummary.style.display = 'block';
    const total = getCartTotal();
    const itemCount = cart.reduce((sum, item) => sum + (item.qty || 1), 0);
    document.getElementById('cart-total').innerHTML = `
      <span class="cart-total-label">${itemCount} item${itemCount > 1 ? 's' : ''}</span>
      <span class="amount">$${total.toFixed(2)}</span>
    `;
  }
}

// ================================
// COMPLETE PURCHASE
// ================================
async function completePurchase() {
  const total = getCartTotal();

  if (window.opEcomTracker) {
    try {
      await window.opEcomTracker.trackPurchase(total);
    } catch (err) {
      console.error('Tracking failed', err);
    }
  }

  clearCart();
  setTimeout(() => {
    window.location.href = '/success';
  }, 500);
}

// ================================
// GET PRODUCT
// ================================
function getProduct(id) {
  return products[id];
}

// ================================
// SHOW LOADING SKELETONS
// ================================
function showSkeletons(container, count = 6) {
  container.innerHTML = Array(count).fill('').map(() => `
    <div class="product-card skeleton-card">
      <div class="skeleton skeleton-image"></div>
      <div class="skeleton skeleton-text" style="width:70%;margin:1rem auto 0.5rem;"></div>
      <div class="skeleton skeleton-text" style="width:40%;margin:0 auto 1rem;"></div>
      <div class="skeleton skeleton-text" style="width:60%;margin:0 auto;"></div>
    </div>
  `).join('');
}

// ================================
// RENDER PRODUCTS
// ================================
function renderProducts() {
  const container = document.getElementById('product-list');
  if (!container) return;

  // Show skeletons first
  showSkeletons(container);

  // Simulate brief load then render real products
  setTimeout(() => {
    container.innerHTML = Object.values(products).map((product, i) => `
      <div class="product-card" style="animation-delay: ${i * 0.08}s">
        <div class="product-image-container">
          <img src="${product.image}" alt="${product.name}" class="product-img" loading="lazy">
          <span class="product-category">${product.category}</span>
        </div>
        <div class="product-name">${product.name}</div>
        <div class="product-price">$${product.price.toFixed(2)}</div>
        <div class="product-actions">
          <a href="/product?id=${product.id}" class="btn btn-secondary btn-sm">View Details</a>
          <button class="btn btn-primary btn-sm" onclick='addToCart(${JSON.stringify(product)})'>Add to Cart</button>
        </div>
      </div>
    `).join('');
  }, 400);
}

// ================================
// RENDER PRODUCT DETAIL
// ================================
function renderProductDetail() {
  const container = document.getElementById('product-detail');
  if (!container) return;

  const urlParams = new URLSearchParams(window.location.search);
  const productId = urlParams.get('id');
  const product = getProduct(productId);

  if (!product) {
    container.innerHTML = `
      <div class="section-header">
        <h2>Product not found</h2>
        <p>The product you're looking for doesn't exist.</p>
        <a href="/products" class="btn btn-primary" style="margin-top:1rem;">Back to Products</a>
      </div>`;
    return;
  }

  container.innerHTML = `
    <div class="product-detail-card">
      <div class="product-detail-image">
        <img src="${product.image}" alt="${product.name}" class="detail-img">
      </div>
      <div class="product-detail-info">
        <span class="product-category-badge">${product.category}</span>
        <h1>${product.name}</h1>
        <p class="price">$${product.price.toFixed(2)}</p>
        <p class="description">${product.description}</p>
        <button class="btn btn-primary" onclick="addToCart(${JSON.stringify(product).replace(/"/g, '&quot;')})">
          Add to Cart
        </button>
        <a href="/products" class="btn btn-secondary" style="margin-left:0.5rem;">Back to Products</a>
      </div>
    </div>
  `;
}

// ================================
// PAGE TRANSITION
// ================================
function initPageTransition() {
  document.body.classList.add('page-enter');

  // Intercept navigation for smooth transitions
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href]');
    if (!link) return;

    const href = link.getAttribute('href');
    // Only handle internal links
    if (!href || href.startsWith('http') || href.startsWith('#') || href.startsWith('javascript')) return;

    e.preventDefault();
    document.body.classList.add('page-exit');
    setTimeout(() => {
      window.location.href = href;
    }, 200);
  });
}

// ================================
// INIT
// ================================
document.addEventListener('DOMContentLoaded', () => {
  updateCartCount();
  renderCart();
  renderProducts();
  renderProductDetail();
  initPageTransition();
});

// Expose globally
window.addToCart = addToCart;
window.removeFromCart = removeFromCart;
window.changeQty = changeQty;
window.completePurchase = completePurchase;
