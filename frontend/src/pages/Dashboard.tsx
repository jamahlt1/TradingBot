import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Dashboard: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Dashboard Overview
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>Summary statistics and charts will appear here.</Typography>
    </Paper>
  </Box>
);

export default Dashboard;