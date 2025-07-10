import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders dashboard navigation', () => {
  render(<App />);
  expect(screen.getByText(/Trading Bot Dashboard/i)).toBeInTheDocument();
  expect(screen.getByText(/Dashboard/i)).toBeInTheDocument();
  expect(screen.getByText(/Strategies/i)).toBeInTheDocument();
  expect(screen.getByText(/Accounts/i)).toBeInTheDocument();
  expect(screen.getByText(/Trades/i)).toBeInTheDocument();
  expect(screen.getByText(/Analytics/i)).toBeInTheDocument();
  expect(screen.getByText(/Settings/i)).toBeInTheDocument();
});
