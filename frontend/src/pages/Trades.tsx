import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Trades: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Trades
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>Trade history and execution UI will appear here.</Typography>
    </Paper>
  </Box>
);

export default Trades;