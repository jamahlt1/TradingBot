import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Analytics: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Analytics
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>Analytics and charting UI will appear here.</Typography>
    </Paper>
  </Box>
);

export default Analytics;