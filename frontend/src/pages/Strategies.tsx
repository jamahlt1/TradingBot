import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Strategies: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Strategies
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>Strategy management UI will appear here.</Typography>
    </Paper>
  </Box>
);

export default Strategies;