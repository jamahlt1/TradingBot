import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Accounts: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Accounts
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>Account management UI will appear here.</Typography>
    </Paper>
  </Box>
);

export default Accounts;