import React, { useState, useEffect } from 'react';
import { 
  Container, Row, Col, Card, Button, Form, 
  Table, Badge, Modal, Alert, Spinner 
} from 'react-bootstrap';

interface Strategy {
  id: number;
  name: string;
  strategy_type: string;
  description?: string;
  allocation: number;
  risk_per_trade: number;
  rr_target: number;
  trailing_stop?: number;
  active: boolean;
  created_at: string;
}

const Strategies: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    strategy_type: 'pairs_trading',
    description: '',
    allocation: 0.1,
    risk_per_trade: 0.02,
    rr_target: 2.0,
    trailing_stop: 0.0,
    active: true
  });

  const strategyTypes = [
    { value: 'pairs_trading', label: 'Pairs Trading' },
    { value: 'scalping', label: 'Scalping' },
    { value: 'swing_trading', label: 'Swing Trading' },
    { value: 'trend_trading', label: 'Trend Trading' },
    { value: 'sentiment_trading', label: 'Sentiment Trading' },
    { value: 'news_trading', label: 'News Trading' },
    { value: 'crypto_trend', label: 'Crypto Trend' },
    { value: 'straddle_hedging', label: 'Straddle Hedging' },
    { value: 'stat_arb', label: 'Statistical Arbitrage' },
    { value: 'ict_concepts', label: 'ICT Concepts' }
  ];

  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      const response = await fetch('/api/strategies/');
      if (response.ok) {
        const data = await response.json();
        setStrategies(data);
      }
    } catch (error) {
      console.error('Error fetching strategies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/strategies/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      if (response.ok) {
        setShowModal(false);
        fetchStrategies();
        setFormData({
          name: '',
          strategy_type: 'pairs_trading',
          description: '',
          allocation: 0.1,
          risk_per_trade: 0.02,
          rr_target: 2.0,
          trailing_stop: 0.0,
          active: true
        });
      }
    } catch (error) {
      console.error('Error creating strategy:', error);
    }
  };

  const handleRunStrategy = async (strategyId: number) => {
    try {
      const response = await fetch(`/api/strategies/${strategyId}/run`, {
        method: 'POST',
      });
      if (response.ok) {
        const result = await response.json();
        alert(`Strategy executed! Signals: ${result.signals.length}`);
      }
    } catch (error) {
      console.error('Error running strategy:', error);
    }
  };

  const handleBacktest = async (strategyId: number) => {
    try {
      const response = await fetch(`/api/strategies/${strategyId}/backtest?start_date=2024-01-01&end_date=2024-12-31`, {
        method: 'POST',
      });
      if (response.ok) {
        const result = await response.json();
        alert(`Backtest completed! Total return: ${result.total_return}%`);
      }
    } catch (error) {
      console.error('Error backtesting strategy:', error);
    }
  };

  if (loading) {
    return (
      <Container className="mt-4">
        <div className="text-center">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <Row className="mb-4">
        <Col>
          <h2>Strategy Management</h2>
          <p className="text-muted">Create and manage your trading strategies</p>
        </Col>
        <Col xs="auto">
          <Button variant="primary" onClick={() => setShowModal(true)}>
            <i className="fas fa-plus me-2"></i>
            New Strategy
          </Button>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card>
            <Card.Header>
              <h5 className="mb-0">Active Strategies</h5>
            </Card.Header>
            <Card.Body>
              {strategies.length === 0 ? (
                <Alert variant="info">
                  No strategies created yet. Create your first strategy to get started!
                </Alert>
              ) : (
                <Table responsive striped>
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Allocation</th>
                      <th>Risk</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strategies.map((strategy) => (
                      <tr key={strategy.id}>
                        <td>{strategy.name}</td>
                        <td>
                          <Badge bg="secondary">
                            {strategyTypes.find(t => t.value === strategy.strategy_type)?.label || strategy.strategy_type}
                          </Badge>
                        </td>
                        <td>{(strategy.allocation * 100).toFixed(1)}%</td>
                        <td>{(strategy.risk_per_trade * 100).toFixed(1)}%</td>
                        <td>
                          <Badge bg={strategy.active ? "success" : "secondary"}>
                            {strategy.active ? "Active" : "Inactive"}
                          </Badge>
                        </td>
                        <td>
                          <Button
                            size="sm"
                            variant="outline-primary"
                            className="me-2"
                            onClick={() => handleRunStrategy(strategy.id)}
                          >
                            Run
                          </Button>
                          <Button
                            size="sm"
                            variant="outline-info"
                            onClick={() => handleBacktest(strategy.id)}
                          >
                            Backtest
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Create/Edit Strategy Modal */}
      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Create New Strategy</Modal.Title>
        </Modal.Header>
        <Form onSubmit={handleSubmit}>
          <Modal.Body>
            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Strategy Name</Form.Label>
                  <Form.Control
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    required
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Strategy Type</Form.Label>
                  <Form.Select
                    value={formData.strategy_type}
                    onChange={(e) => setFormData({...formData, strategy_type: e.target.value})}
                  >
                    {strategyTypes.map((type) => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>

            <Form.Group className="mb-3">
              <Form.Label>Description</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
              />
            </Form.Group>

            <Row>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Allocation (%)</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="100"
                    value={formData.allocation * 100}
                    onChange={(e) => setFormData({...formData, allocation: parseFloat(e.target.value) / 100})}
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Risk per Trade (%)</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    max="10"
                    value={formData.risk_per_trade * 100}
                    onChange={(e) => setFormData({...formData, risk_per_trade: parseFloat(e.target.value) / 100})}
                  />
                </Form.Group>
              </Col>
              <Col md={4}>
                <Form.Group className="mb-3">
                  <Form.Label>Risk/Reward Target</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={formData.rr_target}
                    onChange={(e) => setFormData({...formData, rr_target: parseFloat(e.target.value)})}
                  />
                </Form.Group>
              </Col>
            </Row>

            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>Trailing Stop (%)</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.1"
                    min="0"
                    value={formData.trailing_stop * 100}
                    onChange={(e) => setFormData({...formData, trailing_stop: parseFloat(e.target.value) / 100})}
                  />
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Check
                    type="checkbox"
                    label="Active"
                    checked={formData.active}
                    onChange={(e) => setFormData({...formData, active: e.target.checked})}
                  />
                </Form.Group>
              </Col>
            </Row>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowModal(false)}>
              Cancel
            </Button>
            <Button variant="primary" type="submit">
              Create Strategy
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    </Container>
  );
};

export default Strategies;