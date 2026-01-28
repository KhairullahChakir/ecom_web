/**
 * ShopDemo - Main JavaScript
 * Cart management and tracker integration
 */

// Cart state
let cart = JSON.parse(localStorage.getItem('shopDemo_cart') || '[]');

// Update cart count in navbar
function updateCartCount() {
  const countEl = document.getElementById('cart-count');
  if (countEl) {
    countEl.textContent = cart.length;
  }
}

// Add to cart
function addToCart(product) {
  cart.push(product);
  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();

  // Track event
  if (window.opEcomTracker) {
    window.opEcomTracker.trackEvent('add_to_cart', 'ecommerce', product.name, product.price);
  }

  alert(`${product.name} added to cart!`);
}

// Remove from cart
function removeFromCart(index) {
  const removed = cart.splice(index, 1)[0];
  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();

  if (window.opEcomTracker) {
    window.opEcomTracker.trackEvent('remove_from_cart', 'ecommerce', removed.name);
  }

  renderCart();
}

// Clear cart
function clearCart() {
  cart = [];
  localStorage.setItem('shopDemo_cart', JSON.stringify(cart));
  updateCartCount();
}

// Get cart total
function getCartTotal() {
  return cart.reduce((sum, item) => sum + item.price, 0);
}

// Render cart items (for cart page)
function renderCart() {
  const cartContainer = document.getElementById('cart-items');
  const cartSummary = document.getElementById('cart-summary');

  if (!cartContainer) return;

  if (cart.length === 0) {
    cartContainer.innerHTML = `
      <div class="empty-cart">
        <p>Your cart is empty</p>
        <a href="/products.html" class="btn btn-primary">Browse Products</a>
      </div>
    `;
    if (cartSummary) cartSummary.style.display = 'none';
    return;
  }

  cartContainer.innerHTML = cart.map((item, index) => `
    <div class="cart-item">
      <div class="cart-item-image">${item.emoji}</div>
      <div class="cart-item-details">
        <div class="cart-item-name">${item.name}</div>
        <div class="cart-item-price">$${item.price.toFixed(2)}</div>
      </div>
      <button class="btn btn-secondary" onclick="removeFromCart(${index})">Remove</button>
    </div>
  `).join('');

  if (cartSummary) {
    cartSummary.style.display = 'block';
    document.getElementById('cart-total').textContent = `$${getCartTotal().toFixed(2)}`;
  }
}

// Complete purchase
function completePurchase() {
  const total = getCartTotal();

  // Track purchase with tracker
  if (window.opEcomTracker) {
    window.opEcomTracker.trackPurchase(total);
  }

  // Clear cart
  clearCart();

  // Redirect to success page
  window.location.href = '/success.html';
}

// Product data
const products = {
  1: { id: 1, name: 'Running Shoes', price: 99.99, emoji: '👟', pageValue: 25 },
  2: { id: 2, name: 'Wireless Headphones', price: 149.99, emoji: '🎧', pageValue: 35 },
  3: { id: 3, name: 'Smart Watch', price: 299.99, emoji: '⌚', pageValue: 50 },
  4: { id: 4, name: 'Laptop Bag', price: 79.99, emoji: '💼', pageValue: 20 },
  5: { id: 5, name: 'Bluetooth Speaker', price: 129.99, emoji: '🔊', pageValue: 30 },
  6: { id: 6, name: 'Fitness Tracker', price: 89.99, emoji: '📱', pageValue: 28 }
};

// Get product by ID
function getProduct(id) {
  return products[id];
}

// Render product detail
function renderProductDetail() {
  const container = document.getElementById('product-detail');
  if (!container) return;

  const urlParams = new URLSearchParams(window.location.search);
  const productId = urlParams.get('id');
  const product = getProduct(productId);

  if (!product) {
    container.innerHTML = '<p>Product not found</p>';
    return;
  }

  container.innerHTML = `
    <div class="product-detail-card">
      <div class="product-detail-image">${product.emoji}</div>
      <div class="product-detail-info">
        <h1>${product.name}</h1>
        <p class="price">$${product.price.toFixed(2)}</p>
        <p class="description">Premium quality product with excellent reviews. Perfect for everyday use.</p>
        <button class="btn btn-success" onclick="addToCart(${JSON.stringify(product).replace(/"/g, '&quot;')})">
          Add to Cart
        </button>
      </div>
    </div>
  `;
}

// Render all products
function renderProducts() {
  const container = document.getElementById('products-grid');
  if (!container) return;

  container.innerHTML = Object.values(products).map(product => `
    <div class="product-card" data-product-id="${product.id}" data-page-value="${product.pageValue}">
      <div class="product-image">${product.emoji}</div>
      <h3>${product.name}</h3>
      <p class="price">$${product.price.toFixed(2)}</p>
      <div class="product-actions">
        <a href="/product.html?id=${product.id}" class="btn btn-secondary">View</a>
        <button class="btn btn-success" onclick='addToCart(${JSON.stringify(product)})'>Add</button>
      </div>
    </div>
  `).join('');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  updateCartCount();
  renderCart();
  renderProducts();
  renderProductDetail();
});

// Expose functions globally
window.addToCart = addToCart;
window.removeFromCart = removeFromCart;
window.completePurchase = completePurchase;
