import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                products: resolve(__dirname, 'products.html'),
                about: resolve(__dirname, 'about.html'),
                cart: resolve(__dirname, 'cart.html'),
                checkout: resolve(__dirname, 'checkout.html'),
                product: resolve(__dirname, 'product.html'),
                success: resolve(__dirname, 'success.html'),
                analytics: resolve(__dirname, 'analytics.html'),
            },
        },
    },
});
